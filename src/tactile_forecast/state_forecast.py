"""v1 forecaster over the explicit physical state s(t) (pour + slice).

Loads the analytic physical-state trajectories (data/actionsense_states/state_N.npy, shape
(T, C, 6) = per-hand [F, xbar, ybar, sxx, syy, sxy]), z-normalizes per feature, windows into
(past -> future), and provides:
  - baselines: persistence, linear-velocity, local-linear-fit (the "ramp" structured model)
  - a small GRU seq2seq (learned dynamics)
Skill is reported per physical variable vs persistence (1 - MSE/MSE_persistence), in original units.

Numpy for data/baselines; torch only for the GRU (installed CPU-only locally).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

FEATS = ("F", "xbar", "ybar", "sxx", "syy", "sxy")  # per hand, from physical_state


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_manifest(root):
    return [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]


def load_trajectories(root, label_substr):
    """Return list of (T, C*6) float32 trajectories whose label contains label_substr."""
    trajs = []
    for r in load_manifest(root):
        if label_substr.lower() in r["label"].lower():
            st = np.load(os.path.join(root, f"state_{r['idx']}.npy"))  # (T, C, 6)
            trajs.append(st.reshape(st.shape[0], -1).astype(np.float32))
    return trajs


def feature_names(n_hands):
    return [f"{h}_{f}" for h in range(n_hands) for f in FEATS]


class Normalizer:
    def __init__(self, trajs):
        X = np.concatenate(trajs, 0)
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6

    def fwd(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean


def make_windows(trajs, t_in, t_out, stride):
    """-> X:(N,t_in,D), Y:(N,t_out,D) in RAW units (normalize later)."""
    Xs, Ys = [], []
    win = t_in + t_out
    for tr in trajs:
        for s in range(0, tr.shape[0] - win + 1, stride):
            Xs.append(tr[s:s + t_in])
            Ys.append(tr[s + t_in:s + win])
    if not Xs:
        return np.zeros((0, t_in, 0)), np.zeros((0, t_out, 0))
    return np.stack(Xs), np.stack(Ys)


def split_trajectories(trajs, val_frac=0.2, seed=0):
    """Split by TRAJECTORY (not window) to avoid leakage."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    n_val = max(1, int(round(len(trajs) * val_frac)))
    val = set(idx[:n_val].tolist())
    tr = [t for i, t in enumerate(trajs) if i not in val]
    va = [t for i, t in enumerate(trajs) if i in val]
    return tr, va


# --------------------------------------------------------------------------- #
# Baselines (numpy, in RAW units) — return (N, t_out, D)
# --------------------------------------------------------------------------- #
def bl_persistence(X, t_out):
    last = X[:, -1:, :]
    return np.repeat(last, t_out, axis=1)


def bl_velocity(X, t_out):
    v = X[:, -1, :] - X[:, -2, :]
    steps = np.arange(1, t_out + 1)[None, :, None]
    return X[:, -1:, :] + steps * v[:, None, :]


def bl_linfit(X, t_out):
    """Local linear fit over the input window, extrapolated (the 'ramp' model)."""
    N, t_in, D = X.shape
    t = np.arange(t_in)
    tc = t - t.mean()
    denom = (tc * tc).sum() + 1e-9
    # slope/intercept per (N,D)
    slope = (tc[None, :, None] * (X - X.mean(1, keepdims=True))).sum(1) / denom
    intercept = X.mean(1) - slope * t.mean()
    tf = np.arange(t_in, t_in + t_out)[None, :, None]
    return intercept[:, None, :] + slope[:, None, :] * tf


# --------------------------------------------------------------------------- #
# Skill metric (per feature), in RAW units
# --------------------------------------------------------------------------- #
def skill_per_feature(pred, targ, persist):
    """pred/targ/persist: (N,t_out,D). Return dict feature_idx -> skill = 1 - mse/mse_pers."""
    mse = ((pred - targ) ** 2).mean((0, 1))         # (D,)
    mse_p = ((persist - targ) ** 2).mean((0, 1)) + 1e-12
    return 1.0 - mse / mse_p, mse, mse_p


# --------------------------------------------------------------------------- #
# GRU seq2seq (torch)
# --------------------------------------------------------------------------- #
def build_gru(D, hidden=64, layers=1):
    import torch.nn as nn

    class GRUSeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.GRU(D, hidden, layers, batch_first=True)
            self.dec = nn.GRU(D, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, D)

        def forward(self, x, t_out):
            import torch
            _, h = self.enc(x)
            inp = x[:, -1:, :]                       # last observed frame
            outs = []
            for _ in range(t_out):
                o, h = self.dec(inp, h)
                y = self.head(o)                     # predict RESIDUAL (delta) from last
                inp = inp + y
                outs.append(inp)
            return torch.cat(outs, dim=1)

    return GRUSeq2Seq()
