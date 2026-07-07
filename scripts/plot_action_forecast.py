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
    ap.add_argument("--t-out", type=int, default=5, help="forecast horizon (5 = 0.5s at 10 Hz)")
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
    # proper train/TEST split — hold out ~25% of clips (the viz clip is in the test set)
    rng = np.random.default_rng(1)
    order = rng.permutation(len(data))
    n_test = max(2, int(round(0.25 * len(data))))
    test_ids = set(order[:n_test].tolist()) | {viz_i}
    train = [d for i, d in enumerate(data) if i not in test_ids]
    test = [d for i, d in enumerate(data) if i in test_ids]
    vfeat, vtarg, vaid = data[viz_i]
    TGT = ["F_fast", "x_fast", "y_fast"][args.target]
    print(f"train {len(train)} / test {len(test)} clips; visualize TEST clip #{viz_i}, target {TGT}")

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
    X, A, Yin, Y, G = windows(train, args.t_in, args.t_out, 2)
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
            mu, lv = model(xt[b], at[b], yl[b], args.t_out)
            (0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()).backward()
            opt.step()

    # --- TEST-set skill (all held-out clips) vs persistence-of-fast ---
    model.eval()
    Xte, Ate, Yin_te, Yte, _ = windows(test, args.t_in, args.t_out, 2)
    with torch.no_grad():
        mu_te, _ = model(torch.tensor(((Xte - mu_x) / sd_x).astype(np.float32)),
                         torch.tensor(Ate),
                         torch.tensor(((Yin_te[:, -1] - mu_y) / sd_y).astype(np.float32)), args.t_out)
        mu_te = mu_te.numpy() * sd_y + mu_y
    persist_te = np.repeat(Yin_te[:, -1:], args.t_out, 1)
    sk_te = 1 - ((mu_te - Yte) ** 2).mean((0, 1)) / (((persist_te - Yte) ** 2).mean((0, 1)) + 1e-12)
    print(f"TEST-set skill vs persistence: {sk_te.mean():+.3f}  (per target {np.round(sk_te,2)})")

    # HONEST multi-step forecast: at non-overlapping anchors, seed with the true last observed
    # frame then roll t_out steps forward AUTOREGRESSIVELY (model feeds its OWN predictions).
    # This is exactly the task the +CV skill measures. Persistence = repeat the last true value.
    k = args.target
    fn = (vfeat - mu_x) / sd_x
    ho = args.t_out
    seg_t, seg_mu, seg_sd, seg_pers = [], [], [], []
    with torch.no_grad():
        for a in range(args.t_in, vfeat.shape[0] - ho, ho):     # non-overlapping anchors
            x = torch.tensor(fn[a - args.t_in:a][None], dtype=torch.float32)
            y_last = torch.tensor(((vtarg[a - 1] - mu_y) / sd_y)[None], dtype=torch.float32)
            mu, lv = model(x, torch.tensor([vaid]), y_last, ho)   # ho-step autoregressive rollout
            seg_t.append(np.arange(a, a + ho))
            seg_mu.append(mu[0, :, k].numpy() * sd_y[k] + mu_y[k])
            seg_sd.append(np.exp(0.5 * lv[0, :, k].numpy()) * sd_y[k])
            seg_pers.append(np.full(ho, vtarg[a - 1, k]))         # persistence-of-fast
    ts = np.concatenate(seg_t); mus = np.concatenate(seg_mu)
    sds = np.concatenate(seg_sd); pers = np.concatenate(seg_pers)
    trues = vtarg[ts, k]
    sk_clip = 1 - np.mean((mus - trues) ** 2) / (np.mean((pers - trues) ** 2) + 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    tt = np.arange(args.t_in, vfeat.shape[0])
    ax1.plot(tt, vtarg[tt, k], "k-", lw=1.6, label="true (fast component)")
    for i, seg in enumerate(seg_t):                              # each 0.5s forecast segment
        lab = dict(label="forecast mean (%.1fs, autoregressive)" % (ho / 10)) if i == 0 else {}
        ax1.plot(seg, seg_mu[i], "C0-", lw=1.6, **lab)
        ax1.fill_between(seg, seg_mu[i] - 2 * seg_sd[i], seg_mu[i] + 2 * seg_sd[i], color="C0", alpha=0.20,
                         **(dict(label="+/-2 sigma") if i == 0 else {}))
    ax1.plot(ts, pers, color="0.6", ls="--", lw=1.0, label="persistence-of-fast")
    for a in [s[0] for s in seg_t]:
        ax1.axvline(a, color="0.85", lw=0.6)
    ax1.set_ylabel(f"{TGT}"); ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"v2 {ho/10:.1f}s multi-step forecast on TEST clip — {args.viz_action}, {TGT}   "
                  f"(this clip skill {sk_clip:+.2f}; test-set {sk_te.mean():+.2f} vs persistence)")

    lo = float(min((mus - 3 * sds).min(), trues.min()))
    hi = float(max((mus + 3 * sds).max(), trues.max()))
    grid = np.linspace(lo, hi, 200)
    dens = np.exp(-0.5 * ((grid[:, None] - mus[None, :]) / sds[None, :]) ** 2) / (sds[None, :] * np.sqrt(2 * np.pi))
    ax2.imshow(dens, origin="lower", aspect="auto", extent=[ts[0], ts[-1], lo, hi], cmap="magma")
    ax2.plot(ts, trues, "c-", lw=1.2, label="true")
    ax2.set_xlabel("time step (~10 Hz)  |  vertical lines = anchors where forecast restarts from truth")
    ax2.set_ylabel(f"{TGT}"); ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("predicted probability density  p(value | past)  (brighter = higher density)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    covered = np.mean(np.abs(trues - mus) <= 2 * sds)
    print(f"[done] {args.out}  (this clip skill={sk_clip:+.2f}, {covered*100:.0f}% within +/-2 sigma)")


if __name__ == "__main__":
    main()
