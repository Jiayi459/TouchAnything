"""v2 forecaster: probabilistic model of the FAST action component of the physical state.

Motivation (see docs/TACTILE_FORECAST_PLAN.md + SESSION_LOG): the raw physical state is
dominated by a slow grip/postural component that persistence predicts trivially -> no skill
headroom. Here we low/high-pass split F and CoP, MODEL THE FAST (stroke/pour) component (which
decorrelates within ~2 s), pool the four smooth actions with an action embedding, and predict
a DISTRIBUTION (mean + variance) -> calibrated forecast AND the expert 'normal band' for feedback.

    python scripts/train_action_dynamics.py
    python scripts/train_action_dynamics.py --actions Pour,Slice,Peel,Clean --t-in 10 --t-out 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def slow_fast(sig, fps, cut):
    b, a = butter(2, cut / (fps / 2.0), "low")
    slow = filtfilt(b, a, sig, axis=0)
    return slow, sig - slow


def build_features(state, fps, cut, ds):
    """state (T,C,6) -> (feat (T',Din), target (T',3)) for the active hand.
    target = fast [F, x, y]; input = [F_fast,x_fast,y_fast, F_slow, vx, vy]."""
    state = state[::ds]
    fps = fps / ds
    h = int(np.argmax(state[:, :, 0].mean(0)))  # active hand by mean force
    F, x, y = state[:, h, 0], state[:, h, 1], state[:, h, 2]
    Fs, Ff = slow_fast(F, fps, cut)
    xs, xf = slow_fast(x, fps, cut)
    ys, yf = slow_fast(y, fps, cut)
    vx = np.gradient(x) * fps
    vy = np.gradient(y) * fps
    target = np.stack([Ff, xf, yf], 1).astype(np.float32)
    feat = np.stack([Ff, xf, yf, Fs, vx, vy], 1).astype(np.float32)
    return feat, target


def load_pooled(root, action_subs, fps_default, cut, ds):
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    data = []
    for r in rows:
        aid = next((i for i, s in enumerate(action_subs) if s.lower() in r["label"].lower()), None)
        if aid is None:
            continue
        st = np.load(os.path.join(root, f"state_{r['idx']}.npy"))
        feat, targ = build_features(st, r.get("fps", fps_default), cut, ds)
        if feat.shape[0] >= 20:
            data.append((feat, targ, aid))
    return data


def windows(data, t_in, t_out, stride):
    Xs, As, Yin, Ys, gid = [], [], [], [], []
    for gi, (feat, targ, aid) in enumerate(data):
        win = t_in + t_out
        for s in range(0, feat.shape[0] - win + 1, stride):
            Xs.append(feat[s:s + t_in])
            Yin.append(targ[s:s + t_in])
            Ys.append(targ[s + t_in:s + win])
            As.append(aid); gid.append(gi)
    return (np.stack(Xs), np.array(As), np.stack(Yin), np.stack(Ys), np.array(gid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Pour,Slice,Peel,Clean")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--downsample", type=int, default=3)     # 30 -> 10 Hz
    ap.add_argument("--cut", type=float, default=0.4)         # low-pass cutoff Hz (slow/fast split)
    ap.add_argument("--t-in", type=int, default=10)           # 1.0 s
    ap.add_argument("--t-out", type=int, default=5)           # 0.5 s
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    import torch.nn as nn

    subs = [s.strip() for s in args.actions.split(",")]
    data = load_pooled(args.root, subs, args.fps, args.cut, args.downsample)
    counts = {s: sum(1 for d in data if d[2] == i) for i, s in enumerate(subs)}
    print(f"pooled trajectories: {len(data)}  {counts}")
    TGT = ["F_fast", "x_fast", "y_fast"]

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

    X, A, Yin, Y, G = windows(data, args.t_in, args.t_out, args.stride)
    # per-feature normalization from all windows (fast comps ~zero-mean already)
    mu_x = X.reshape(-1, X.shape[-1]).mean(0); sd_x = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    mu_y = Y.reshape(-1, 3).mean(0); sd_y = Y.reshape(-1, 3).std(0) + 1e-6
    Xn = (X - mu_x) / sd_x
    Yn = (Y - mu_y) / sd_y
    Yin_n = (Yin - mu_y) / sd_y

    rng = np.random.default_rng(args.seed)
    fold_of = rng.integers(0, args.folds, size=len(data))  # fold by trajectory
    win_fold = fold_of[G]

    skills, covs = [], []
    for f in range(args.folds):
        tr = win_fold != f; te = win_fold == f
        if te.sum() < 5 or tr.sum() < 20:
            continue
        torch.manual_seed(args.seed)
        model = ProbGRU(X.shape[-1], len(subs), args.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        xt = torch.tensor(Xn[tr]); at = torch.tensor(A[tr]); yl = torch.tensor(Yin_n[tr][:, -1])
        yt = torch.tensor(Yn[tr])
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 64):
                b = perm[i:i + 64]; opt.zero_grad()
                mu, lv = model(xt[b], at[b], yl[b], args.t_out)
                nll = 0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()
                nll.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mu, lv = model(torch.tensor(Xn[te]), torch.tensor(A[te]),
                           torch.tensor(Yin_n[te][:, -1]), args.t_out)
            mu = mu.numpy() * sd_y + mu_y
            sd = np.sqrt(np.exp(lv.numpy())) * sd_y
        Yte = Y[te]
        persist = np.repeat(Yin[te][:, -1:], args.t_out, 1)  # persistence-of-fast
        mse_m = ((mu - Yte) ** 2).mean((0, 1))
        mse_p = ((persist - Yte) ** 2).mean((0, 1)) + 1e-12
        sk = 1 - mse_m / mse_p
        cov = (np.abs(Yte - mu) <= 2 * sd).mean()
        skills.append(sk); covs.append(cov)

    skills = np.array(skills)
    print(f"\n{args.folds}-fold CV — skill vs persistence-of-fast (per target), coverage@2sigma:")
    for j, t in enumerate(TGT):
        print(f"  {t:<8} skill={skills[:,j].mean():+.3f} ± {skills[:,j].std():.3f}")
    print(f"  MEAN     skill={skills.mean():+.3f}   band coverage@2sd={np.mean(covs):.2f} (ideal ~0.95)")


if __name__ == "__main__":
    main()
