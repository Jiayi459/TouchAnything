"""Tactile-MAP input pipeline for F/CoP forecasting.

Per recording, the raw pressure map clip_<idx>.npy (T, 2, 32, 32) is turned into model input:
  1. downsample [::ds]                      (10 Hz; matches the harness target)
  2. causal per-taxel baseline (first N)    base = clip[:N].mean(0); x = clip - base; clip>=0
  3. log1p amplitude compression            (fixed; tames heavy-tailed peaks)
  4. global TRAIN scale (one mean/std over ALL taxels/hands/frames -> same scaling every taxel)

The TARGET is the harness's 6-dim F/CoP (eval_harness.dataset.load_target), z-normed per channel
on TRAIN (eval_harness.dataset.Norm). Windows/origins/split all come from the harness so exported
predictions align 1:1 with evaluate.py --model-preds.

CAUSALITY: the baseline uses only the first N frames (past); windows use only frames <= origin t.
Windows are sliced lazily (a 10 s history over all clips would be tens of GB if materialized).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..eval_harness.config import Config
from ..eval_harness.dataset import Norm, load_target
from ..eval_harness.baselines.base import origins

H_MOMENTS = 3          # F, CoP-x, CoP-y per hand (target has 6 = 2 hands x 3)


def clip_path(cfg: Config, idx: int) -> str:
    return os.path.join(cfg.abspath("states_root"), f"clip_{idx}.npy")


def available_idxs(cfg: Config, idxs: list[int]) -> list[int]:
    """Subset of idxs whose raw map clip_<idx>.npy exists locally (for pre-restream smoke runs)."""
    return [i for i in idxs if os.path.exists(clip_path(cfg, i))]


def load_map(cfg: Config, idx: int, baseline_frames: int) -> np.ndarray:
    """clip_<idx>.npy (T,2,32,32) -> (T',2,32,32) float32: downsample + causal first-N baseline."""
    clip = np.load(clip_path(cfg, idx)).astype(np.float32)[:: cfg.downsample]   # (T',2,32,32)
    n = min(baseline_frames, len(clip))
    base = clip[:n].mean(0, keepdims=True)                                       # per-taxel, past-only
    return np.clip(clip - base, 0.0, None)


def compress(x: np.ndarray, alpha: float) -> np.ndarray:
    """log1p amplitude compression, normalized so compress(1/alpha)~O(1). Fixed (no train stats)."""
    return np.log1p(alpha * np.clip(x, 0.0, None)) / np.log1p(alpha)


@dataclass(frozen=True)
class MapNorm:
    """Global scalar normalization of the compressed map (same scaling for every taxel)."""
    mean: float
    std: float
    alpha: float

    @staticmethod
    def from_train(train_maps: dict[int, np.ndarray], alpha: float) -> "MapNorm":
        vals = np.concatenate([compress(m, alpha).reshape(-1) for m in train_maps.values()])
        return MapNorm(float(vals.mean()), float(vals.std() + 1e-6), alpha)

    def apply(self, m: np.ndarray) -> np.ndarray:
        return ((compress(m, self.alpha) - self.mean) / self.std).astype(np.float32)


def load_raw(cfg: Config, idxs: list[int], baseline_frames: int
             ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """-> (maps: idx->(T',2,32,32) baseline-corrected raw, targets: idx->(T',6))."""
    maps = {i: load_map(cfg, i, baseline_frames) for i in idxs}
    tgts = {i: load_target(cfg, i) for i in idxs}
    # guard: map and target must share the time axis (both downsampled the same way)
    for i in idxs:
        n = min(len(maps[i]), len(tgts[i]))
        maps[i], tgts[i] = maps[i][:n], tgts[i][:n]
    return maps, tgts


def normalize(maps: dict[int, np.ndarray], mnorm: MapNorm) -> dict[int, np.ndarray]:
    return {i: mnorm.apply(m) for i, m in maps.items()}


class MapWindows(Dataset):
    """Lazy rolling-origin windows aligned to the harness origins.

    Returns (X (t_in,2,32,32) normalized map history, Y (H,6) normalized target future). For
    origins with < t_in frames of history, the window is LEFT-padded with zeros (post-baseline
    "no contact") -> a prediction exists at EVERY harness origin (score_external alignment)."""

    def __init__(self, maps_n: dict[int, np.ndarray], tgts_n: dict[int, np.ndarray],
                 cfg: Config, t_in: int):
        self.maps, self.tgts, self.t_in, self.H = maps_n, tgts_n, t_in, cfg.horizon
        self.index = [(i, int(t)) for i in sorted(maps_n) for t in origins(len(maps_n[i]), cfg)]

    def __len__(self):
        return len(self.index)

    def _window(self, i: int, t: int) -> np.ndarray:
        M = self.maps[i]
        win = M[max(t - self.t_in + 1, 0): t + 1]                # (<=t_in, 2,32,32)
        if win.shape[0] < self.t_in:                             # causal left-pad with zeros
            pad = np.zeros((self.t_in - win.shape[0],) + M.shape[1:], np.float32)
            win = np.concatenate([pad, win], 0)
        return win

    def __getitem__(self, k: int):
        i, t = self.index[k]
        x = self._window(i, t)
        y = self.tgts[i][t + 1: t + 1 + self.H]                  # (H,6)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def recording_windows(map_n: np.ndarray, cfg: Config, t_in: int) -> tuple[np.ndarray, np.ndarray]:
    """For export: all (n_origins, t_in, 2,32,32) windows of one recording + the origin indices."""
    ors = origins(len(map_n), cfg)
    ds_ = MapWindows({0: map_n}, {0: np.zeros((len(map_n), 6), np.float32)}, cfg, t_in)
    X = np.stack([ds_._window(0, int(t)) for t in ors]) if len(ors) else np.zeros((0, t_in, 2, 32, 32), np.float32)
    return X, ors
