"""Train one CV fold of a tactile->tactile forecaster.

Examples
--------
# Leave-Trajectory-Out fold 0, ConvGRU, grasp subset:
python -m src.tactile_forecast.train --config configs/tactile/convgru.yaml \
    --protocol lto --fold 0 --scope grasp

# Pretrain on ALL trajectories (no held-out test), save checkpoint:
python -m src.tactile_forecast.train --config configs/tactile/convgru.yaml \
    --scope full --pretrain --out runs/convgru_pretrain

# Fine-tune from a pretrained checkpoint:
python -m src.tactile_forecast.train --config configs/tactile/convgru.yaml \
    --protocol lto --fold 0 --scope grasp --pretrained runs/convgru_pretrain/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import baselines, engine
from . import tactile_utils as U
from .data import TactileWindows, preload_clips, split_train_val
from .models import build_model

DATA_ROOTS = {
    "grasp": os.path.join("datasets", "grasp_hold_lift_tactile"),
    "full": os.path.join("datasets", "EgoTouch"),
}


def resolve_device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def eval_baseline(loader, fn, inv, mask_np, t_out):
    preds, targs, lasts = [], [], []
    for x, y, _ in loader:
        p = fn(x, t_out).numpy()
        preds.append(inv(p)); targs.append(inv(y.numpy())); lasts.append(inv(x[:, -1].numpy()))
    pred = np.concatenate(preds); targ = np.concatenate(targs); last = np.concatenate(lasts)
    m = U.horizon_metrics(pred, targ, last, mask_np)
    return m, float(np.mean([m[h]["skill"] for h in m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", choices=["lto", "loto"], default="lto")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scope", choices=["grasp", "full"], default="grasp")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pretrain", action="store_true", help="train on all data, no test split")
    ap.add_argument("--pretrained", default=None, help="checkpoint to init weights from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=0, help="cap windows for a smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    device = resolve_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data_root = args.data_root or DATA_ROOTS[args.scope]
    out = args.out or os.path.join("runs", f"{cfg['name']}_{args.scope}_{args.protocol}_f{args.fold}")
    os.makedirs(out, exist_ok=True)
    print(f"[cfg] {cfg['name']} | device={device} | scope={args.scope} | root={data_root} | out={out}")

    # --- data ---
    mask = U.load_or_build_mask(data_root, cache=os.path.join(data_root, "sensor_mask.npy"))
    fwd, inv = U.make_transform(cfg.get("transform", "log1p"), cfg.get("alpha", 10.0))
    trajs = U.list_trajectories(data_root)
    tasks = [t for _, t in trajs]
    clips = preload_clips(trajs, mask, fwd)
    print(f"[data] {len(trajs)} trajectories, {sum(c.shape[0] for c in clips)} frames")

    t_in, t_out = cfg["t_in"], cfg["t_out"]
    cfg["in_ch"] = 2

    # --- splits ---
    if args.pretrain:
        all_idx = np.arange(len(trajs))
        tr_idx, va_idx = split_train_val(all_idx, cfg.get("val_frac", 0.1), args.seed)
        te_idx = np.array([], dtype=int)
    else:
        if args.protocol == "lto":
            splits = U.kfold_by_trajectory(len(trajs), cfg.get("kfolds", 5), args.seed)
        else:
            splits = U.leave_one_task_out(tasks)
        assert 0 <= args.fold < len(splits), f"fold {args.fold} out of range ({len(splits)})"
        tr_full, te_idx = splits[args.fold]
        tr_idx, va_idx = split_train_val(tr_full, cfg.get("val_frac", 0.15), args.seed)
    print(f"[split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    def make_loader(indices, train):
        ds = TactileWindows(clips, indices, mask, t_in, t_out, cfg["stride"],
                            augment=train and cfg.get("augment", True),
                            noise=cfg.get("noise", 0.0), scale=cfg.get("scale", 0.0),
                            hflip=cfg.get("hflip", False), seed=args.seed)
        if args.max_windows and len(ds) > args.max_windows:
            ds.index = ds.index[:args.max_windows]
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train,
                          num_workers=args.num_workers, drop_last=train), len(ds)

    train_loader, n_tr = make_loader(tr_idx, True)
    val_loader, n_va = make_loader(va_idx, False)
    print(f"[windows] train={n_tr} val={n_va}")

    # --- model ---
    model = build_model(cfg).to(device)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        print(f"[init] loaded pretrained weights from {args.pretrained}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg['name']} params={n_params/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    ss_epochs = cfg.get("ss_epochs", 0)
    active_w = cfg.get("active_weight", 1.0)

    # --- training with early stopping on val mean-skill ---
    best_skill, best_epoch, patience = -1e9, -1, cfg.get("patience", 15)
    log_path = os.path.join(out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mean_skill"])
    for epoch in range(cfg["epochs"]):
        ssp = engine.ss_schedule(epoch, ss_epochs)
        tr_loss = engine.train_one_epoch(model, train_loader, opt, device, ssp, active_w)
        _, val_skill = engine.evaluate(model, val_loader, device, inv, mask)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{val_skill:.6f}"])
        flag = ""
        if val_skill > best_skill:
            best_skill, best_epoch = val_skill, epoch
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(out, "best.pt"))
            flag = " *"
        print(f"  epoch {epoch:3d} loss={tr_loss:.5f} val_skill={val_skill:.4f}{flag}")
        if epoch - best_epoch >= patience:
            print(f"  early stop (no val improvement for {patience} epochs)")
            break

    # --- final evaluation ---
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device)["model"])
    summary = {"config": cfg, "best_epoch": best_epoch, "val_mean_skill": best_skill,
               "n_params_M": n_params / 1e6, "splits": {"train": int(len(tr_idx)),
               "val": int(len(va_idx)), "test": int(len(te_idx))}}
    if len(te_idx) > 0:
        test_loader, n_te = make_loader(te_idx, False)
        m_model, s_model = engine.evaluate(model, test_loader, device, inv, mask)
        m_pers, s_pers = eval_baseline(test_loader, baselines.persistence, inv, mask, t_out)
        m_vel, s_vel = eval_baseline(test_loader, baselines.last_velocity, inv, mask, t_out)
        summary["test"] = {"model": m_model, "persistence": m_pers, "last_velocity": m_vel,
                           "mean_skill": {"model": s_model, "persistence": s_pers,
                                          "last_velocity": s_vel}}
        # per-horizon CSV
        with open(os.path.join(out, "test_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["horizon", "model_mse", "model_skill", "model_iou", "model_force_mae",
                        "pers_mse", "vel_mse"])
            for h in m_model:
                w.writerow([h, f"{m_model[h]['mse']:.6f}", f"{m_model[h]['skill']:.4f}",
                            f"{m_model[h]['iou']:.4f}", f"{m_model[h]['force_mae']:.4f}",
                            f"{m_pers[h]['mse']:.6f}", f"{m_vel[h]['mse']:.6f}"])
        print(f"[test] model mean-skill={s_model:.4f} (persistence={s_pers:.4f}, "
              f"last_vel={s_vel:.4f}) over {n_te} windows")
        print(f"[test] skill@h: " + ", ".join(f"h{h}={m_model[h]['skill']:.3f}" for h in m_model))

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] artifacts -> {out}")


if __name__ == "__main__":
    main()
