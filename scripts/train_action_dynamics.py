"""Train the v2 action-dynamics forecaster (thin CLI over src/tactile_forecast/action_dynamics.py).

Sweeps one or more PAST-context lengths (--pasts, in seconds), all forecasting the next
--future-sec. For each: k-fold cross-validated skill vs persistence-of-fast, then a final model
trained on all clips and saved as a checkpoint. Training logic lives in the library; this script
only orchestrates + reports + saves.

    python scripts/train_action_dynamics.py --actions Slice,Peel                 # sweep 1/2/3/5/10s
    python scripts/train_action_dynamics.py --actions Slice,Peel --pasts 1        # single 1s model
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tactile_forecast import action_dynamics as AD  # noqa: E402


def cross_validate(data, n_act, t_in, t_out, folds, hidden, epochs, seed):
    """k-fold CV by trajectory -> (skill_per_target (folds,3), coverage list)."""
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(data))
    sks, covs = [], []
    for f in range(folds):
        tr = [d for i, d in enumerate(data) if fold_of[i] != f]
        te = [d for i, d in enumerate(data) if fold_of[i] == f]
        if len(te) < 1 or len(tr) < 3:
            continue
        m, norm = AD.train(tr, n_act, t_in, t_out, hidden=hidden, epochs=epochs, seed=seed)
        sk, _, cov = AD.evaluate(m, norm, te, t_in, t_out)
        sks.append(sk); covs.append(cov)
    return np.array(sks), covs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--downsample", type=int, default=3)          # 30 -> 10 Hz
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--pasts", default="1,2,3,5,10", help="past-context lengths in seconds to sweep")
    ap.add_argument("--future-sec", type=float, default=1.0, help="forecast horizon in seconds")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs", help="where to save per-past checkpoints")
    args = ap.parse_args()

    subs = [s.strip() for s in args.actions.split(",")]
    data = AD.load_pooled(args.root, subs, args.downsample, args.cut)
    counts = {s: sum(1 for d in data if d[2] == i) for i, s in enumerate(subs)}
    fps = 30.0 / args.downsample
    t_out = int(round(args.future_sec * fps))
    pasts = [float(p) for p in args.pasts.split(",")]
    print(f"pooled trajectories: {len(data)}  {counts}")
    print(f"forecast horizon = {t_out} frames ({args.future_sec}s); sweeping pasts (s): {pasts}\n")
    if len(data) < args.folds:
        raise SystemExit("too few trajectories for CV")

    print(f"{'past':>5} {'t_in':>5} | {'F_fast':>16} {'x_fast':>16} {'y_fast':>16} | {'MEAN':>7} {'cov':>5}")
    print("-" * 78)
    summary = []
    for p in pasts:
        t_in = int(round(p * fps))
        sks, covs = cross_validate(data, len(subs), t_in, t_out, args.folds,
                                   args.hidden, args.epochs, args.seed)
        mean = sks.mean(); cov = float(np.mean(covs))
        cells = "  ".join(f"{sks[:, j].mean():+.3f}+/-{sks[:, j].std():.3f}" for j in range(3))
        print(f"{p:>4.0f}s {t_in:>5} | {cells} | {mean:+.3f} {cov:>5.2f}")
        # final model on ALL clips for this past length -> checkpoint
        model, norm = AD.train(data, len(subs), t_in, t_out,
                               hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        out = os.path.join(args.outdir, f"ad_{'-'.join(subs).lower()}_p{p:.0f}s.pt")
        AD.save(out, model, norm, dict(subs=subs, n_act=len(subs), t_in=t_in, t_out=t_out,
                                       cut=args.cut, downsample=args.downsample, hidden=args.hidden))
        summary.append((p, mean, out))

    print("\nSUMMARY  mean skill vs persistence by past-context:")
    for p, mean, out in summary:
        print(f"  past {p:>4.0f}s -> skill {mean:+.3f}   (saved {out})")


if __name__ == "__main__":
    main()
