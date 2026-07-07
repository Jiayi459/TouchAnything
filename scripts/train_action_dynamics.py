"""Train the v2 action-dynamics forecaster (thin CLI over src/tactile_forecast/action_dynamics.py).

Cross-validates skill vs persistence-of-fast, then trains a final model on all clips and saves a
checkpoint (train logic lives in the library; this script only orchestrates + reports + saves).

    python scripts/train_action_dynamics.py --actions Slice,Peel
    python scripts/train_action_dynamics.py --actions Slice,Peel --t-out 10 --out runs/slice_peel.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tactile_forecast import action_dynamics as AD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--downsample", type=int, default=3)     # 30 -> 10 Hz
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--t-in", type=int, default=10)          # 1.0 s past
    ap.add_argument("--t-out", type=int, default=5)          # 0.5 s future
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="checkpoint path (default runs/ad_<actions>.pt)")
    args = ap.parse_args()

    subs = [s.strip() for s in args.actions.split(",")]
    data = AD.load_pooled(args.root, subs, args.downsample, args.cut)
    counts = {s: sum(1 for d in data if d[2] == i) for i, s in enumerate(subs)}
    print(f"pooled trajectories: {len(data)}  {counts}")
    if len(data) < args.folds:
        raise SystemExit("too few trajectories for CV")

    # --- k-fold CV by trajectory ---
    rng = np.random.default_rng(args.seed)
    fold_of = rng.integers(0, args.folds, size=len(data))
    sks, covs = [], []
    for f in range(args.folds):
        tr = [d for i, d in enumerate(data) if fold_of[i] != f]
        te = [d for i, d in enumerate(data) if fold_of[i] == f]
        if len(te) < 1 or len(tr) < 3:
            continue
        m, norm = AD.train(tr, len(subs), args.t_in, args.t_out,
                           hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        sk, _, cov = AD.evaluate(m, norm, te, args.t_in, args.t_out)
        sks.append(sk); covs.append(cov)
    sks = np.array(sks)
    print(f"\n{args.folds}-fold CV — skill vs persistence-of-fast (per target), coverage@2sd:")
    for j, t in enumerate(AD.TARGETS):
        print(f"  {t:<8} skill={sks[:, j].mean():+.3f} +/- {sks[:, j].std():.3f}")
    print(f"  MEAN     skill={sks.mean():+.3f}   band coverage@2sd={np.mean(covs):.2f} (ideal ~0.95)")

    # --- train final model on ALL clips, save checkpoint ---
    model, norm = AD.train(data, len(subs), args.t_in, args.t_out,
                           hidden=args.hidden, epochs=args.epochs, seed=args.seed)
    out = args.out or os.path.join("runs", f"ad_{'-'.join(subs).lower()}.pt")
    meta = dict(subs=subs, n_act=len(subs), t_in=args.t_in, t_out=args.t_out,
                cut=args.cut, downsample=args.downsample, hidden=args.hidden)
    AD.save(out, model, norm, meta)
    print(f"[saved] final model -> {out}")


if __name__ == "__main__":
    main()
