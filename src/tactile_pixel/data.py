"""Torch dataset for windowed tactile->tactile forecasting.

Builds sliding windows (t_in -> t_out) from in-memory clips. Splits are made at the
*trajectory* level (no window leakage). Augmentations are mask-safe.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import tactile_utils as U


def preload_clips(trajs, mask, fwd):
    """Load every clip once (transform applied, masked). Returns list of (T,2,21,21)."""
    return [U.load_clip(p, mask, fwd) for p, _ in trajs]


class TactileWindows(Dataset):
    def __init__(self, clips, indices, mask, t_in, t_out, stride,
                 augment=False, noise=0.0, scale=0.0, hflip=False, seed=0):
        self.clips = clips
        self.mask = mask.astype(np.float32)
        self.t_in, self.t_out = t_in, t_out
        self.augment = augment
        self.noise, self.scale, self.hflip = noise, scale, hflip
        self.rng = np.random.default_rng(seed)
        lengths = [clips[i].shape[0] for i in indices]
        local = U.build_window_index(lengths, t_in, t_out, stride)
        # map local sequence position back to global clip index
        self.index = [(indices[si], s) for (si, s) in local]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ci, s = self.index[k]
        win = self.clips[ci][s:s + self.t_in + self.t_out]
        x = win[:self.t_in].copy()
        y = win[self.t_in:].copy()
        mask = self.mask.copy()
        if self.augment:
            if self.scale > 0:
                f = 1.0 + float(self.rng.uniform(-self.scale, self.scale))
                x *= f; y *= f
            if self.noise > 0:
                x = x + (x > 0) * self.rng.normal(0, self.noise, x.shape).astype(np.float32)
            if self.hflip and self.rng.random() < 0.5:
                x = x[..., ::-1].copy(); y = y[..., ::-1].copy(); mask = mask[..., ::-1].copy()
        x = np.clip(x, 0.0, 1.0) * mask[None]
        y = np.clip(y, 0.0, 1.0) * mask[None]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def split_train_val(train_idx, val_frac, seed=0):
    """Hold out a fraction of *trajectories* from train_idx for early stopping."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(train_idx)
    n_val = max(1, int(round(len(order) * val_frac)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])
