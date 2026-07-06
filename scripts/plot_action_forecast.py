"""Visualize the v2 probabilistic forecaster: mean +/- band and the probability-density map.

Trains the ProbGRU on the pooled clips (holding out one clip), then rolls a 1-step-ahead
probabilistic forecast across the held-out clip and plots, for the fast force component:
  (top)    true trajectory vs predicted mean with +/-1 sigma and +/-2 sigma bands
  (bottom) the predicted Gaussian probability DENSITY at each time step (heatmap), true overlaid

    python scripts/plot_action_forecast.py --viz-action Slice
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_action_dynamics import load_pooled, windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Pour,Slice")
    ap.add_argument("--viz-action", default="Slice", help="which action's clip to visualize")
    ap.add_argument("--target", type=int, default=0, help="0=F_fast 1=x_fast 2=y_fast")
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--t-in", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default="docs/action_forecast_density.png")
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s.strip() for s in args.actions.split(",")]
    data = load_pooled(args.root, subs, 30.0, args.cut, args.downsample)
    viz_i = next(i for i, d in enumerate(data) if d[2] == subs.index(
        next(s for s in subs if s.lower() in args.viz_action.lower() or args.viz_action.lower() in s.lower())))
    train = [d for i, d in enumerate(data) if i != viz_i]
    vfeat, vtarg, vaid = data[viz_i]
    TGT = ["F_fast", "x_fast", "y_fast"][args.target]
    print(f"train on {len(train)} clips, visualize clip #{viz_i} (action id {vaid}), target {TGT}")

    class ProbGRU(nn.Module):
        def __init__(self, din, n_act, hid):
            super().__init__()
            self.emb = nn.Embedding(n_act, 8)
            self.enc = nn.GRU(din, hid, batch_first=True)
            self.dec = nn.GRU(3, hid, batch_first=True)
            self.mu = nn.Linear(hid + 8, 3)
            self.lv = nn.Linear(hid + 8, 3)

        def forward(self, x, aid, y_last, t_out):
            _, h = self.enc(x)
            e = self.emb(aid)
            inp = y_last.unsqueeze(1)
            mus, lvs = [], []
            for _ in range(t_out):
                o, h = self.dec(inp, h)
                oc = torch.cat([o[:, -1], e], -1)
                mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
                mus.append(mu); lvs.append(lv)
                inp = mu.unsqueeze(1)
            return torch.stack(mus, 1), torch.stack(lvs, 1)

    # windows from training clips, normalize
    X, A, Yin, Y, G = windows(train, args.t_in, 5, 2)
    mu_x = X.reshape(-1, X.shape[-1]).mean(0); sd_x = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    mu_y = Y.reshape(-1, 3).mean(0); sd_y = Y.reshape(-1, 3).std(0) + 1e-6
    Xn = ((X - mu_x) / sd_x).astype(np.float32)
    Yn = ((Y - mu_y) / sd_y).astype(np.float32)
    Yin_n = ((Yin - mu_y) / sd_y).astype(np.float32)

    torch.manual_seed(0)
    model = ProbGRU(X.shape[-1], len(subs), args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    xt = torch.tensor(Xn); at = torch.tensor(A); yl = torch.tensor(Yin_n[:, -1]); yt = torch.tensor(Yn)
    for ep in range(args.epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            mu, lv = model(xt[b], at[b], yl[b], 5)
            (0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()).backward()
            opt.step()

    # rolling 1-step-ahead forecast across the held-out clip
    model.eval()
    fn = (vfeat - mu_x) / sd_x
    tn = (vtarg - mu_y) / sd_y
    ts, mus, sds, trues = [], [], [], []
    with torch.no_grad():
        for t in range(args.t_in, vfeat.shape[0]):
            x = torch.tensor(fn[t - args.t_in:t][None], dtype=torch.float32)
            yl1 = torch.tensor(tn[t - 1][None], dtype=torch.float32)
            mu, lv = model(x, torch.tensor([vaid]), yl1, 1)
            k = args.target
            mus.append(float(mu[0, 0, k]) * sd_y[k] + mu_y[k])
            sds.append(float(np.exp(0.5 * lv[0, 0, k].item())) * sd_y[k])
            trues.append(float(vtarg[t, k])); ts.append(t)
    ts = np.array(ts); mus = np.array(mus); sds = np.array(sds); trues = np.array(trues)

    # figure: band + density heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(ts, trues, "k-", lw=1.8, label="true (fast component)")
    ax1.plot(ts, mus, "C0-", lw=1.5, label="predicted mean")
    ax1.fill_between(ts, mus - 2 * sds, mus + 2 * sds, color="C0", alpha=0.18, label="+/-2 sigma")
    ax1.fill_between(ts, mus - sds, mus + sds, color="C0", alpha=0.30, label="+/-1 sigma")
    ax1.set_ylabel(f"{TGT}"); ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"v2 probabilistic 1-step forecast — {args.viz_action} (held-out clip), {TGT}")

    # density map: Gaussian pdf per time column
    lo = float(min((mus - 3 * sds).min(), trues.min()))
    hi = float(max((mus + 3 * sds).max(), trues.max()))
    grid = np.linspace(lo, hi, 200)
    dens = np.exp(-0.5 * ((grid[:, None] - mus[None, :]) / sds[None, :]) ** 2) / (sds[None, :] * np.sqrt(2 * np.pi))
    ax2.imshow(dens, origin="lower", aspect="auto",
               extent=[ts[0], ts[-1], lo, hi], cmap="magma")
    ax2.plot(ts, trues, "c-", lw=1.2, label="true")
    ax2.set_xlabel("time step (~10 Hz)"); ax2.set_ylabel(f"{TGT}")
    ax2.set_title("predicted probability density  p(value | past)  (brighter = higher density)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    covered = np.mean(np.abs(trues - mus) <= 2 * sds)
    print(f"[done] {args.out}  (this clip: {covered*100:.0f}% of true within +/-2 sigma)")


if __name__ == "__main__":
    main()
