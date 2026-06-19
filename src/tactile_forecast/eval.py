"""Evaluate a saved checkpoint on a CV fold's test split (model vs baselines).

python -m src.tactile_forecast.eval --ckpt runs/convgru_grasp_lto_f0/best.pt \
    --protocol lto --fold 0 --scope grasp
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import baselines, engine
from . import tactile_utils as U
from .data import TactileWindows, preload_clips
from .models import build_model
from .train import DATA_ROOTS, eval_baseline, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--protocol", choices=["lto", "loto"], default="lto")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scope", choices=["grasp", "full"], default="grasp")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(args.device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = sd["cfg"]
    cfg["in_ch"] = 2
    data_root = args.data_root or DATA_ROOTS[args.scope]

    mask = U.load_or_build_mask(data_root, cache=os.path.join(data_root, "sensor_mask.npy"))
    fwd, inv = U.make_transform(cfg.get("transform", "log1p"), cfg.get("alpha", 10.0))
    trajs = U.list_trajectories(data_root)
    tasks = [t for _, t in trajs]
    clips = preload_clips(trajs, mask, fwd)

    if args.protocol == "lto":
        splits = U.kfold_by_trajectory(len(trajs), cfg.get("kfolds", 5), args.seed)
    else:
        splits = U.leave_one_task_out(tasks)
    _, te_idx = splits[args.fold]

    ds = TactileWindows(clips, te_idx, mask, cfg["t_in"], cfg["t_out"], cfg["stride"], augment=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)

    model = build_model(cfg).to(device)
    model.load_state_dict(sd["model"])

    m_model, s_model = engine.evaluate(model, loader, device, inv, mask)
    m_pers, s_pers = eval_baseline(loader, baselines.persistence, inv, mask, cfg["t_out"])
    print(f"test windows={len(ds)}  model mean-skill={s_model:.4f}  persistence={s_pers:.4f}")
    print("h : model_skill  model_iou  model_mse  pers_mse")
    for h in m_model:
        print(f"{h:2d}: {m_model[h]['skill']:+.3f}      {m_model[h]['iou']:.3f}      "
              f"{m_model[h]['mse']:.5f}  {m_pers[h]['mse']:.5f}")


if __name__ == "__main__":
    main()
