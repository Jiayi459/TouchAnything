"""Train + evaluate the v2 action-dynamics forecaster (thin CLI over action_dynamics.py).

Sweeps input representation x hand x past-context; forecasts the next --future-sec of the FAST
target. For each (input_mode, hand): a history x channel skill table + a history x forecast-step
table, and a checkpoint per past. All breakdowns (input_mode, hand, history, forecast-step, channel)
are written to a CSV. Run scripts/check_leakage.py first.

    python scripts/train_action_dynamics.py --actions Slice,Peel
    python scripts/train_action_dynamics.py --actions Slice,Peel --input-modes highpass --hands left
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tactile_forecast import action_dynamics as AD  # noqa: E402


def cross_validate(data, n_act, t_in, t_out, folds, hidden, epochs, seed):
    """-> skt (folds,3), sks (folds,t_out,3), covs (list). Fold by trajectory."""
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(data))
    skt, sks, covs = [], [], []
    for f in range(folds):
        tr = [d for i, d in enumerate(data) if fold_of[i] != f]
        te = [d for i, d in enumerate(data) if fold_of[i] == f]
        if len(te) < 1 or len(tr) < 3:
            continue
        m, norm = AD.train(tr, n_act, t_in, t_out, hidden=hidden, epochs=epochs, seed=seed)
        sk_t, sk_s, _, cov = AD.evaluate(m, norm, te, t_in, t_out)
        skt.append(sk_t); sks.append(sk_s); covs.append(cov)
    return np.array(skt), np.array(sks), covs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--input-modes", default="raw,highpass", help="comma list: raw and/or highpass")
    ap.add_argument("--hands", default="left,right", help="comma list: left and/or right")
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--warmup-sec", type=float, default=5.0)
    ap.add_argument("--pasts", default="1,2,3,5,10", help="past-context lengths (s)")
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--csv", default="docs/action_dynamics_results.csv")
    args = ap.parse_args()

    subs = [s.strip() for s in args.actions.split(",")]
    modes = [m.strip() for m in args.input_modes.split(",")]
    hands = [h.strip() for h in args.hands.split(",")]
    fps = 30.0 / args.downsample
    t_out = int(round(args.future_sec * fps))
    pasts = [float(p) for p in args.pasts.split(",")]
    n_act = len(subs)
    print(f"actions={subs}  input_modes={modes}  hands={hands}  future={t_out}f ({args.future_sec}s)"
          f"  warmup={args.warmup_sec}s  pasts(s)={pasts}\n")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    fcsv = open(args.csv, "w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["input_mode", "hand", "history_s", "forecast_step_s",
                "F_skill", "x_skill", "y_skill", "mean_skill", "coverage"])

    for mode in modes:
        for hand in hands:
            data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                                  input_mode=mode, hand=hand, warmup_sec=args.warmup_sec)
            counts = {s: sum(1 for d in data if d[2] == i) for i, s in enumerate(subs)}
            din = len(AD.feats_for(mode))
            print(f"===== input={mode}  hand={hand}  ({len(data)} clips {counts}, D={din}) =====")
            print(f"{'past':>5} {'t_in':>5} | {'F':>7} {'x':>7} {'y':>7} | {'MEAN':>7} {'cov':>5}")
            print("-" * 52)
            per_step_rows = []  # (past, mean-channel skill per step)
            for p in pasts:
                t_in = int(round(p * fps))
                skt, sks, covs = cross_validate(data, n_act, t_in, t_out, args.folds,
                                                args.hidden, args.epochs, args.seed)
                skt_m = skt.mean(0); sks_m = sks.mean(0); cov = float(np.mean(covs))
                print(f"{p:>4.0f}s {t_in:>5} | {skt_m[0]:>+7.3f} {skt_m[1]:>+7.3f} {skt_m[2]:>+7.3f}"
                      f" | {skt_m.mean():>+7.3f} {cov:>5.2f}")
                per_step_rows.append((p, sks_m.mean(1)))       # (t_out,) mean-channel per step
                for st in range(t_out):
                    w.writerow([mode, hand, p, round((st + 1) / fps, 2),
                                f"{sks_m[st,0]:.4f}", f"{sks_m[st,1]:.4f}", f"{sks_m[st,2]:.4f}",
                                f"{sks_m[st].mean():.4f}", f"{cov:.3f}"])
                # final model on ALL clips -> checkpoint
                model, norm = AD.train(data, n_act, t_in, t_out,
                                       hidden=args.hidden, epochs=args.epochs, seed=args.seed)
                out = os.path.join(args.outdir, f"ad_{'-'.join(subs).lower()}_{mode}_{hand}_p{p:.0f}s.pt")
                AD.save(out, model, norm, dict(subs=subs, n_act=n_act, t_in=t_in, t_out=t_out,
                        cut=args.cut, downsample=args.downsample, hidden=args.hidden,
                        input_mode=mode, hand=hand, din=din))
            # per-forecast-step table (mean channel skill; shows horizon decay within the 1s)
            steps = [f"+{(s+1)/fps:.1f}s" for s in range(t_out)]
            print(f"\n  per-step MEAN skill:  {'past':>5} | " + " ".join(f"{s:>6}" for s in steps))
            for p, arr in per_step_rows:
                print(f"                        {p:>4.0f}s | " + " ".join(f"{v:>+6.2f}" for v in arr))
            print()
    fcsv.close()
    print(f"[csv] full breakdown -> {args.csv}")


if __name__ == "__main__":
    main()
