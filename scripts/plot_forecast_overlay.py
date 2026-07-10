"""Overlay the real tactile signal with the 1 s forecasts of the 5 history-length models.

For 2 held-out TEST clips x both hands x 3 channels (F, CoP-x, CoP-y): plot the true fast signal
plus the rolling 1 s forecast of each of the 5 past-context models (1/2/3/5/10 s) on the same axes
(6 lines/panel). All models forecast 1 s; they differ only in how much past they see (so a longer-
history model's forecast starts later in the clip). Saved to docs/forecast_overlay.png
(results_summary.png is left untouched).

    python scripts/plot_forecast_overlay.py --actions Slice,Peel --input-mode raw
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tactile_forecast import action_dynamics as AD  # noqa: E402

CH = ["F (force)", "CoP-x (stroke)", "CoP-y"]


def forecast_all(model, norm, clip, t_in, t_out):
    """Rolling non-overlapping 1 s forecast, all 3 channels -> (ts, mu (len,3))."""
    import torch
    feat, targ, aid = clip
    fn = norm.nx(feat)
    ts, mus = [], []
    model.eval()
    with torch.no_grad():
        for a in range(t_in, feat.shape[0] - t_out, t_out):
            x = torch.tensor(fn[a - t_in:a][None])
            yl = torch.tensor(norm.ny(targ[a - 1])[None])
            mu, _ = model(x, torch.tensor([aid]), yl, t_out)
            ts.append(np.arange(a, a + t_out)); mus.append(norm.dy(mu[0].numpy()))
    return np.concatenate(ts), np.concatenate(mus, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--input-mode", default="raw")
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--pasts", default="1,2,3,5,10")
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-clips", type=int, default=2, help="how many test clips to show")
    ap.add_argument("--out", default="docs/forecast_overlay.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s.strip() for s in args.actions.split(",")]
    fps = 30.0 / args.downsample
    t_out = int(round(args.future_sec * fps))
    pasts = [float(p) for p in args.pasts.split(",")]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(pasts)))

    # same train/test split + clip indices for both hands (split is by clip index, fixed seed)
    ref = AD.load_pooled(args.root, subs, args.downsample, args.cut, input_mode=args.input_mode, hand="left")
    _, test_ids = AD.split_train_test(len(ref), seed=1)
    # choose the n longest test clips (need >= max_t_in + t_out frames)
    max_tin = int(round(max(pasts) * fps))
    viz_ids = [i for i in test_ids if ref[i][0].shape[0] >= max_tin + t_out][:args.n_clips]
    print(f"visualizing test clips {viz_ids} (both hands); training {len(pasts)} history models/hand")

    # train per hand, forecast the viz clips
    results = {}   # (hand, clip_idx) -> (true (T,3), {past: (ts, mu(len,3))})
    for hand in ["left", "right"]:
        data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                              input_mode=args.input_mode, hand=hand)
        train = [d for i, d in enumerate(data) if i not in test_ids]
        models = {}
        for p in pasts:
            t_in = int(round(p * fps))
            models[p] = AD.train(train, len(subs), t_in, t_out, epochs=args.epochs)
        for ci in viz_ids:
            preds = {}
            for p in pasts:
                m, norm = models[p]
                preds[p] = forecast_all(m, norm, data[ci], int(round(p * fps)), t_out)
            results[(hand, ci)] = (data[ci][1], preds)
        print(f"  hand={hand} done")

    # grid: rows = (clip, hand), cols = 3 channels
    rows = [(ci, hand) for ci in viz_ids for hand in ["left", "right"]]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 3.0 * len(rows)), squeeze=False)
    for ri, (ci, hand) in enumerate(rows):
        true, preds = results[(hand, ci)]
        for k in range(3):
            ax = axes[ri, k]
            ax.plot(np.arange(true.shape[0]), true[:, k], "k-", lw=1.6,
                    label="real" if ri == 0 and k == 0 else None)
            for pi, p in enumerate(pasts):
                ts, mu = preds[p]
                ax.plot(ts, mu[:, k], "-", color=colors[pi], lw=1.1, alpha=0.85,
                        label=f"{p:.0f}s hist" if ri == 0 and k == 0 else None)
            if k == 0:
                ax.set_ylabel(f"clip {ci} / {hand}")
            if ri == 0:
                ax.set_title(CH[k])
            if ri == len(rows) - 1:
                ax.set_xlabel("time step (~10 Hz)")
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle(f"Real tactile vs 1 s forecasts of 5 history models — {args.input_mode} input "
                 f"({args.actions})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
