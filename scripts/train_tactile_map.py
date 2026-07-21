"""Train the tactile-map -> F/CoP forecaster sweep (flatten vs CNN x history) + export preds.

Thin CLI over src/actionsense/tactile_map. For each (encoder, history) it trains on the frozen
harness TRAIN split, early-stops on VAL, and exports TEST predictions as an .npz that the frozen
harness scores:  python -m src.actionsense.eval_harness.evaluate --model-preds <npz> --model-name <n>

    python scripts/train_tactile_map.py                 # full run (needs all 75 maps)
    python scripts/train_tactile_map.py --available --epochs 3   # local smoke on cached maps
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.config import load_config          # noqa: E402
from src.actionsense.tactile_map import train as T                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-config", default="configs/actionsense/tactile_map.yaml")
    ap.add_argument("--available", action="store_true",
                    help="restrict to recordings whose maps are cached locally (pre-restream smoke)")
    ap.add_argument("--encoders", default=None, help="override sweep encoders, comma list")
    ap.add_argument("--histories", default=None, help="override histories (s), comma list")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()                                  # frozen harness cfg (rate/horizon/split/target)
    tmc = yaml.safe_load(open(args.tm_config))
    tm = {**tmc["preprocess"], **tmc["model"], **tmc["optim"]}
    if args.epochs:
        tm["epochs"] = args.epochs
    encoders = args.encoders.split(",") if args.encoders else tmc["sweep"]["encoders"]
    histories = [float(h) for h in args.histories.split(",")] if args.histories \
        else tmc["sweep"]["histories_s"]
    out_dir = tmc["paths"]["out_dir"]
    fps = cfg.fps
    print(f"harness: fps={fps:.0f} horizon={cfg.horizon}  encoders={encoders} histories(s)={histories}"
          f"  {'[AVAILABLE-ONLY smoke]' if args.available else '[full split]'}\n")

    for hist in histories:
        t_in = int(round(hist * fps))
        tr, va, mnorm, tnorm, te = T.build_datasets(cfg, tm, t_in, use_available=args.available)
        print(f"--- history {hist:.0f}s (t_in={t_in})  train_windows={len(tr)} val={len(va)} "
              f"test_recordings={len(te)} ---")
        for enc in encoders:
            model, vmse = T.train(tr, va, cfg, enc, tm, seed=tm["seed"])
            preds = T.export(model, mnorm, tnorm, cfg, tm, t_in, te)
            npz = os.path.join(out_dir, f"preds_{enc}_{hist:.0f}s.npz")
            T.save_preds(npz, preds)
            n_or = sum(p.shape[0] for p in preds.values())
            print(f"  {enc:8}  val_mse={vmse:.4f}  -> {npz}  ({len(preds)} recs, {n_or} origins)")
    print(f"\n[done] score each with: python -m src.actionsense.eval_harness.evaluate "
          f"--model-preds {out_dir}/preds_<enc>_<hist>s.npz --model-name <enc>_<hist>s")


if __name__ == "__main__":
    main()
