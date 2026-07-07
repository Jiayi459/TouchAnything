"""Visualize how much PAST context helps the v2 forecaster predict 1 s of the future.

Trains one probabilistic model per past-window length (1/2/3/5/10 s) — all forecasting the
next 1 s of the fast action component — and plots, for a held-out TEST clip, each model's
HONEST multi-step autoregressive forecast (seeded once with the true last frame, then rolled
on its own predictions) against the truth and the persistence-of-fast baseline.

    python scripts/plot_action_forecast.py --actions Slice,Peel --viz-action Slice --target 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_action_dynamics import load_pooled, windows  # noqa: E402


# ===== MODEL (same architecture as train_action_dynamics.py, at module scope) ============= #
def make_model(din, n_act, hid):
    import torch
    import torch.nn as nn

    class ProbGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_act, 8)
            self.enc = nn.GRU(din, hid, batch_first=True)     # summarize the past window
            self.dec = nn.GRU(3, hid, batch_first=True)       # roll the future forward
            self.mu = nn.Linear(hid + 8, 3)                   # predicted mean per step
            self.lv = nn.Linear(hid + 8, 3)                   # predicted log-variance per step

        def forward(self, x, aid, y_last, t_out):
            _, h = self.enc(x)                                # h = encoding of the past
            e = self.emb(aid)                                 # action embedding
            inp = y_last.unsqueeze(1)                         # decoder seed = last observed target
            mus, lvs = [], []
            for _ in range(t_out):                            # autoregressive rollout
                o, h = self.dec(inp, h)
                oc = torch.cat([o[:, -1], e], -1)
                mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
                mus.append(mu); lvs.append(lv)
                inp = mu.unsqueeze(1)                         # feed the model's OWN prediction back
            return torch.stack(mus, 1), torch.stack(lvs, 1)

    return ProbGRU()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--viz-action", default="Slice")
    ap.add_argument("--target", type=int, default=0, help="0=F_fast 1=x_fast 2=y_fast")
    ap.add_argument("--downsample", type=int, default=3)          # 30 Hz -> 10 Hz
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--pasts", default="1,2,3,5,10", help="past-context lengths in seconds")
    ap.add_argument("--future-sec", type=float, default=1.0, help="forecast horizon in seconds")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default="docs/action_forecast_density.png")
    args = ap.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = 30.0 / args.downsample                                  # effective sample rate (10 Hz)
    t_out = int(round(args.future_sec * fps))                    # future = 1 s -> 10 frames
    pasts = [float(p) for p in args.pasts.split(",")]
    t_ins = [int(round(p * fps)) for p in pasts]                 # 1/2/3/5/10 s -> 10/20/30/50/100
    subs = [s.strip() for s in args.actions.split(",")]
    TGT = ["F_fast", "x_fast", "y_fast"][args.target]

    # ===== SECTION 1: load data + train/TEST split (viz clip is held out) ================= #
    data = load_pooled(args.root, subs, 30.0, args.cut, args.downsample)
    viz_i = next(i for i, d in enumerate(data)
                 if subs[d[2]].lower().startswith(args.viz_action.lower())
                 or args.viz_action.lower().startswith(subs[d[2]].lower()))
    rng = np.random.default_rng(1)
    order = rng.permutation(len(data))
    n_test = max(2, int(round(0.25 * len(data))))
    test_ids = set(order[:n_test].tolist()) | {viz_i}            # ensure viz clip is in TEST
    train = [d for i, d in enumerate(data) if i not in test_ids]
    test = [d for i, d in enumerate(data) if i in test_ids]
    vfeat, vtarg, vaid = data[viz_i]
    print(f"train {len(train)} / test {len(test)} clips; viz TEST clip #{viz_i}; "
          f"future={t_out} frames ({args.future_sec}s); pasts(frames)={t_ins}")

    # normalization from ALL training frames (per-feature; independent of window length)
    allX = np.concatenate([f for f, _, _ in train]); allY = np.concatenate([t for _, t, _ in train])
    mu_x, sd_x = allX.mean(0), allX.std(0) + 1e-6
    mu_y, sd_y = allY.mean(0), allY.std(0) + 1e-6
    k = args.target

    # ===== SECTION 2: helpers — train one model, and forecast a clip ===================== #
    def train_model(t_in):
        X, A, Yin, Y, _ = windows(train, t_in, t_out, 2)         # sliding (past->future) windows
        Xn = ((X - mu_x) / sd_x).astype(np.float32)
        Yn = ((Y - mu_y) / sd_y).astype(np.float32)
        Yl = ((Yin[:, -1] - mu_y) / sd_y).astype(np.float32)     # true last observed target (seed)
        torch.manual_seed(0)
        m = make_model(X.shape[-1], len(subs), args.hidden)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        xt, at, yl, yt = map(torch.tensor, (Xn, A, Yl, Yn))
        for _ in range(args.epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 64):
                b = perm[i:i + 64]; opt.zero_grad()
                mu, lv = m(xt[b], at[b], yl[b], t_out)
                (0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()).backward()  # Gaussian NLL
                opt.step()
        return m

    def test_skill(m, t_in):                                      # mean skill over ALL test clips
        X, A, Yin, Y, _ = windows(test, t_in, t_out, 2)
        with torch.no_grad():
            mu, _ = m(torch.tensor(((X - mu_x) / sd_x).astype(np.float32)), torch.tensor(A),
                      torch.tensor(((Yin[:, -1] - mu_y) / sd_y).astype(np.float32)), t_out)
        mu = mu.numpy() * sd_y + mu_y
        pers = np.repeat(Yin[:, -1:], t_out, 1)                   # persistence-of-fast baseline
        sk = 1 - ((mu - Y) ** 2).mean((0, 1)) / (((pers - Y) ** 2).mean((0, 1)) + 1e-12)
        return sk.mean()

    def forecast_clip(m, t_in):                                   # honest multi-step on the viz clip
        fn = (vfeat - mu_x) / sd_x
        seg = []
        with torch.no_grad():
            for a in range(t_in, vfeat.shape[0] - t_out, t_out):  # non-overlapping anchors
                x = torch.tensor(fn[a - t_in:a][None], dtype=torch.float32)
                yl = torch.tensor(((vtarg[a - 1] - mu_y) / sd_y)[None], dtype=torch.float32)
                mu, lv = m(x, torch.tensor([vaid]), yl, t_out)    # seed once, roll t_out on own preds
                t = np.arange(a, a + t_out)
                seg.append((t, mu[0, :, k].numpy() * sd_y[k] + mu_y[k],
                            np.exp(0.5 * lv[0, :, k].numpy()) * sd_y[k],
                            np.full(t_out, vtarg[a - 1, k])))      # persistence for this segment
        ts = np.concatenate([s[0] for s in seg]); mus = np.concatenate([s[1] for s in seg])
        sds = np.concatenate([s[2] for s in seg]); pers = np.concatenate([s[3] for s in seg])
        trues = vtarg[ts, k]
        sk = 1 - np.mean((mus - trues) ** 2) / (np.mean((pers - trues) ** 2) + 1e-12)
        return seg, ts, mus, sds, pers, trues, sk

    # ===== SECTION 3: sweep past-context lengths; train, evaluate, forecast ============== #
    results = []
    for p, t_in in zip(pasts, t_ins):
        m = train_model(t_in)
        sk_test = test_skill(m, t_in)
        fc = forecast_clip(m, t_in)
        results.append((p, t_in, sk_test, fc))
        print(f"  past {p:>4.1f}s (t_in={t_in:3d}) -> 1s forecast: "
              f"test-set skill={sk_test:+.3f}, viz-clip skill={fc[6]:+.3f}")

    # ===== SECTION 4: plot one row per past-context length =============================== #
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.1 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    tt = np.arange(min(r[1] for r in results), vfeat.shape[0])
    for ax, (p, t_in, sk_test, (seg, ts, _mus, _sds, pers, _trues, sk_clip)) in zip(axes, results):
        ax.plot(tt, vtarg[tt, k], "k-", lw=1.4, label="true")
        for i, (t, mu, sd, _pe) in enumerate(seg):               # each 1 s forecast segment
            ax.plot(t, mu, "C0-", lw=1.5, label="1s forecast (autoregressive)" if i == 0 else None)
            ax.fill_between(t, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.20,
                            label="+/-2 sigma" if i == 0 else None)
        ax.plot(ts, pers, color="0.6", ls="--", lw=0.9, label="persistence-of-fast")
        for a in [s[0][0] for s in seg]:
            ax.axvline(a, color="0.9", lw=0.5)
        ax.set_ylabel(f"{TGT}")
        ax.set_title(f"past {p:.0f}s  ->  forecast next 1s   |   test-set skill {sk_test:+.2f}, "
                     f"this-clip {sk_clip:+.2f}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("time step (~10 Hz)   |   grey vertical lines = anchors (forecast restarts from truth)")
    fig.suptitle(f"How much PAST helps forecast the next 1 s — {args.viz_action}, {TGT} (held-out test clip)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[done] {args.out}")
    print("SUMMARY test-set skill vs past-context:  " +
          "  ".join(f"{p:.0f}s={r[2]:+.2f}" for p, r in zip(pasts, results)))


if __name__ == "__main__":
    main()
