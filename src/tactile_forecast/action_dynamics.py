"""v2 physical-state action-dynamics forecaster — the LIBRARY (import, don't run).

Single source of truth for the model + data + training + forecasting used by the thin CLIs
scripts/train_action_dynamics.py (train -> checkpoint) and scripts/plot_action_forecast.py
(load checkpoint / train -> plot). See docs/TACTILE_FORECAST_PLAN.md.

Idea: split each physical-state signal (force, center-of-pressure) into a slow (grip/postural)
and a fast (stroke/pour) component; forecast the FAST component of the active hand with a
probabilistic GRU (mean + variance). Baseline = persistence-of-fast.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

FEATS = ("F_fast", "x_fast", "y_fast", "F_slow", "vx", "vy")   # per-frame model input (6)
TARGETS = ("F_fast", "x_fast", "y_fast")                        # per-frame model output (3)


# --------------------------------------------------------------------------- #
# Signal processing + features
# --------------------------------------------------------------------------- #
def slow_fast(sig, fps, cut):
    """Low/high-pass split: slow = zero-phase Butterworth low-pass; fast = sig - slow."""
    b, a = butter(2, cut / (fps / 2.0), "low")
    slow = filtfilt(b, a, sig, axis=0)
    return slow, sig - slow


def build_features(state, fps, cut, ds):
    """state (T,C,6) physical moments -> (feat (T',6), target (T',3)) for the ACTIVE hand."""
    state = state[::ds]
    fps = fps / ds
    h = int(np.argmax(state[:, :, 0].mean(0)))          # active hand = highest mean force
    F, x, y = state[:, h, 0], state[:, h, 1], state[:, h, 2]
    Fs, Ff = slow_fast(F, fps, cut)
    _, xf = slow_fast(x, fps, cut)
    _, yf = slow_fast(y, fps, cut)
    vx, vy = np.gradient(x) * fps, np.gradient(y) * fps
    target = np.stack([Ff, xf, yf], 1).astype(np.float32)
    feat = np.stack([Ff, xf, yf, Fs, vx, vy], 1).astype(np.float32)
    return feat, target


def load_pooled(root, action_subs, ds, cut, fps_default=30.0, min_len=20):
    """Read the state dataset -> list of (feat, target, action_id). Matches a clip to an
    action if its label STARTS WITH the action string (so 'Slice' != 'Spread ... bread slice')."""
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    data = []
    for r in rows:
        aid = next((i for i, s in enumerate(action_subs)
                    if r["label"].lower().startswith(s.lower())), None)
        if aid is None:
            continue
        st = np.load(os.path.join(root, f"state_{r['idx']}.npy"))
        feat, targ = build_features(st, r.get("fps", fps_default), cut, ds)
        if feat.shape[0] >= min_len:
            data.append((feat, targ, aid))
    return data


def windows(clips, t_in, t_out, stride):
    """Sliding (past t_in -> future t_out) windows within each clip (no cross-clip leakage)."""
    Xs, As, Yin, Ys, gid = [], [], [], [], []
    for gi, (feat, targ, aid) in enumerate(clips):
        win = t_in + t_out
        for s in range(0, feat.shape[0] - win + 1, stride):
            Xs.append(feat[s:s + t_in]); Yin.append(targ[s:s + t_in])
            Ys.append(targ[s + t_in:s + win]); As.append(aid); gid.append(gi)
    if not Xs:
        return (np.zeros((0, t_in, len(FEATS)), np.float32), np.zeros(0, int),
                np.zeros((0, t_in, 3), np.float32), np.zeros((0, t_out, 3), np.float32), np.zeros(0, int))
    return np.stack(Xs), np.array(As), np.stack(Yin), np.stack(Ys), np.array(gid)


def split_train_test(n, frac=0.25, seed=1, force_test=()):
    """Return (train_ids, test_ids) over clip indices; force_test clips are put in test."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(2, int(round(frac * n)))
    test = set(order[:n_test].tolist()) | set(force_test)
    return [i for i in range(n) if i not in test], sorted(test)


# --------------------------------------------------------------------------- #
# Normalization + model
# --------------------------------------------------------------------------- #
@dataclass
class Norm:
    mu_x: np.ndarray; sd_x: np.ndarray; mu_y: np.ndarray; sd_y: np.ndarray

    @classmethod
    def from_clips(cls, clips):
        X = np.concatenate([f for f, _, _ in clips]); Y = np.concatenate([t for _, t, _ in clips])
        return cls(X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6)

    def nx(self, x): return ((x - self.mu_x) / self.sd_x).astype(np.float32)
    def ny(self, y): return ((y - self.mu_y) / self.sd_y).astype(np.float32)
    def dy(self, y): return y * self.sd_y + self.mu_y


class ProbGRU(nn.Module):
    def __init__(self, din, n_act, hid):
        super().__init__()
        self.emb = nn.Embedding(n_act, 8)
        self.enc = nn.GRU(din, hid, batch_first=True)     # summarize the past window
        self.dec = nn.GRU(3, hid, batch_first=True)       # roll the future forward
        self.mu = nn.Linear(hid + 8, 3)                   # predicted mean per step
        self.lv = nn.Linear(hid + 8, 3)                   # predicted log-variance per step

    def forward(self, x, aid, y_last, t_out):
        _, h = self.enc(x)
        e = self.emb(aid)
        inp = y_last.unsqueeze(1)                         # seed = last observed target
        mus, lvs = [], []
        for _ in range(t_out):
            o, h = self.dec(inp, h)
            oc = torch.cat([o[:, -1], e], -1)
            mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
            mus.append(mu); lvs.append(lv)
            inp = mu.unsqueeze(1)                         # feed model's OWN prediction back
        return torch.stack(mus, 1), torch.stack(lvs, 1)


# --------------------------------------------------------------------------- #
# Train / evaluate / forecast
# --------------------------------------------------------------------------- #
def train(clips, n_act, t_in, t_out, norm=None, hidden=48, epochs=80, lr=3e-3, seed=0):
    """Train a ProbGRU on `clips`. Returns (model, norm)."""
    norm = norm or Norm.from_clips(clips)
    X, A, Yin, Y, _ = windows(clips, t_in, t_out, 2)
    xt = torch.tensor(norm.nx(X)); at = torch.tensor(A)
    yl = torch.tensor(norm.ny(Yin)[:, -1]); yt = torch.tensor(norm.ny(Y))
    torch.manual_seed(seed)
    m = ProbGRU(X.shape[-1], n_act, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            mu, lv = m(xt[b], at[b], yl[b], t_out)
            (0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()).backward()   # Gaussian NLL
            opt.step()
    return m, norm


def evaluate(model, norm, clips, t_in, t_out):
    """Skill (per target + mean) vs persistence-of-fast, and band coverage@2sd, on `clips`."""
    X, A, Yin, Y, _ = windows(clips, t_in, t_out, 2)
    model.eval()
    with torch.no_grad():
        mu, lv = model(torch.tensor(norm.nx(X)), torch.tensor(A),
                       torch.tensor(norm.ny(Yin)[:, -1]), t_out)
    mu = norm.dy(mu.numpy()); sd = np.sqrt(np.exp(lv.numpy())) * norm.sd_y
    pers = np.repeat(Yin[:, -1:], t_out, 1)
    sk = 1 - ((mu - Y) ** 2).mean((0, 1)) / (((pers - Y) ** 2).mean((0, 1)) + 1e-12)
    cov = float((np.abs(Y - mu) <= 2 * sd).mean())
    return sk, float(sk.mean()), cov


def forecast_clip(model, norm, clip, t_in, t_out, target=0):
    """Honest multi-step forecast on one clip: at non-overlapping anchors, seed once with the
    true last frame then roll t_out steps on the model's OWN predictions. Returns a dict."""
    feat, targ, aid = clip
    k = target
    fn = norm.nx(feat)
    model.eval()
    seg = []
    with torch.no_grad():
        for a in range(t_in, feat.shape[0] - t_out, t_out):
            x = torch.tensor(fn[a - t_in:a][None])
            yl = torch.tensor(norm.ny(targ[a - 1])[None])
            mu, lv = model(x, torch.tensor([aid]), yl, t_out)
            t = np.arange(a, a + t_out)
            seg.append((t, norm.dy(mu[0].numpy())[:, k],
                        np.sqrt(np.exp(lv[0, :, k].numpy())) * norm.sd_y[k],
                        np.full(t_out, targ[a - 1, k])))
    ts = np.concatenate([s[0] for s in seg]); mus = np.concatenate([s[1] for s in seg])
    sds = np.concatenate([s[2] for s in seg]); pers = np.concatenate([s[3] for s in seg])
    trues = targ[ts, k]
    sk = 1 - np.mean((mus - trues) ** 2) / (np.mean((pers - trues) ** 2) + 1e-12)
    return dict(segments=seg, ts=ts, mu=mus, sd=sds, pers=pers, true=trues, skill=float(sk))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save(path, model, norm, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "norm": (norm.mu_x, norm.sd_x, norm.mu_y, norm.sd_y),
                "meta": meta}, path)


def load(path):
    """Return (model, norm, meta)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    m = ProbGRU(len(FEATS), meta["n_act"], meta["hidden"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, Norm(*ck["norm"]), meta
