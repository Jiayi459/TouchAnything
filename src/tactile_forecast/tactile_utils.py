"""Framework-agnostic (numpy-only) utilities for tactile->tactile forecasting.

Kept free of torch so the data/mask/split/metric logic is unit-testable without a
GPU/torch install (training runs on the ND CRC CUDA env). See TACTILE_PREDICTION_PLAN.md.

Conventions
-----------
A tactile clip is an array of shape (T, 2, 21, 21): T frames, 2 hand channels
(0=left, 1=right), 21x21 grid. ~50.8% of cells are a *structural* sensor mask
(always-NaN); valid taxels = 217 per hand. NaNs are zero-filled; losses/metrics are
computed only over valid taxels.
"""
from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np

GRID = 21
N_HANDS = 2
HANDS = ("left", "right")


# --------------------------------------------------------------------------- #
# Sensor mask
# --------------------------------------------------------------------------- #
def compute_sensor_mask(npz_path: str) -> np.ndarray:
    """Per-hand validity mask (2, 21, 21) bool from one trajectory.

    The NaN pattern is structural (identical across frames/trajectories), so any
    single clip suffices: a cell is valid iff it is finite in frame 0.
    """
    z = np.load(npz_path)
    out = np.zeros((N_HANDS, GRID, GRID), dtype=bool)
    for i, h in enumerate(HANDS):
        out[i] = np.isfinite(z[f"{h}_pressure_grid"][0])
    return out


def load_or_build_mask(data_root: str, cache: str | None = None) -> np.ndarray:
    """Load cached mask or build it from the first npz under data_root."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    files = sorted(glob.glob(os.path.join(data_root, "*", "*", "pressure_grids.npz")))
    if not files:  # full-dataset layout: scene/task/traj
        files = sorted(glob.glob(os.path.join(data_root, "*", "*", "*", "pressure_grids.npz")))
    if not files:
        raise FileNotFoundError(f"No pressure_grids.npz under {data_root}")
    mask = compute_sensor_mask(files[0])
    if cache:
        np.save(cache, mask)
    return mask


# --------------------------------------------------------------------------- #
# Amplitude transforms (data is normalized to [0, 1]; sparse, heavy-tailed)
# --------------------------------------------------------------------------- #
def make_transform(name: str, alpha: float = 10.0):
    """Return (fwd, inv) amplitude transforms mapping ~[0,1] -> ~[0,1]."""
    if name == "raw":
        return (lambda x: x, lambda x: x)
    if name == "sqrt":
        return (lambda x: np.sqrt(np.clip(x, 0, None)),
                lambda x: np.clip(x, 0, None) ** 2)
    if name == "log1p":
        c = np.log1p(alpha)
        return (lambda x: np.log1p(alpha * np.clip(x, 0, None)) / c,
                lambda x: np.expm1(np.clip(x, 0, None) * c) / alpha)
    raise ValueError(f"unknown transform {name!r}")


# --------------------------------------------------------------------------- #
# Sequence loading
# --------------------------------------------------------------------------- #
def load_clip(npz_path: str, mask: np.ndarray,
              fwd: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """Load one clip as (T, 2, 21, 21) float32: NaN->0, masked cells zeroed, fwd applied."""
    z = np.load(npz_path)
    L = np.nan_to_num(z["left_pressure_grid"], nan=0.0)
    R = np.nan_to_num(z["right_pressure_grid"], nan=0.0)
    clip = np.stack([L, R], axis=1).astype(np.float32)  # (T,2,21,21)
    if fwd is not None:
        clip = fwd(clip).astype(np.float32)
    clip *= mask[None]  # zero structurally-invalid cells
    return clip


def list_trajectories(data_root: str):
    """Return list of (npz_path, task_name). Supports both subset (task/traj) and
    full (scene/task/traj) layouts."""
    out = []
    files = sorted(glob.glob(os.path.join(data_root, "*", "*", "pressure_grids.npz")))
    if files:  # task/traj
        for f in files:
            task = os.path.basename(os.path.dirname(os.path.dirname(f)))
            out.append((f, task))
        return out
    files = sorted(glob.glob(os.path.join(data_root, "*", "*", "*", "pressure_grids.npz")))
    for f in files:  # scene/task/traj
        task = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out.append((f, task))
    return out


# --------------------------------------------------------------------------- #
# Window index + splits
# --------------------------------------------------------------------------- #
def build_window_index(lengths, t_in: int, t_out: int, stride: int):
    """List of (seq_idx, start) s.t. a full [start, start+t_in+t_out) window fits."""
    win = t_in + t_out
    idx = []
    for si, T in enumerate(lengths):
        last = T - win
        if last < 0:
            continue
        for s in range(0, last + 1, stride):
            idx.append((si, s))
    return idx


def kfold_by_trajectory(n_seq: int, k: int, seed: int = 0):
    """Leave-Trajectory-Out k-fold: list of (train_idx, test_idx) over sequence indices."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_seq)
    folds = np.array_split(order, k)
    splits = []
    for i in range(k):
        test = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        splits.append((train, test))
    return splits


def leave_one_task_out(tasks):
    """Leave-One-Task-Out: list of (train_idx, test_idx); one fold per unique task."""
    tasks = list(tasks)
    uniq = sorted(set(tasks))
    splits = []
    for t in uniq:
        test = np.array([i for i, x in enumerate(tasks) if x == t])
        train = np.array([i for i, x in enumerate(tasks) if x != t])
        splits.append((train, test))
    return splits


# --------------------------------------------------------------------------- #
# Numpy metrics (computed in ORIGINAL [0,1] space, masked, per-horizon)
# --------------------------------------------------------------------------- #
def _masked(arr, mask):
    """arr:(...,2,21,21) -> values at valid taxels, broadcasting mask.

    Accepts a bool or 0/1 (float/int) mask; cast to bool for boolean indexing.
    """
    m = np.broadcast_to(np.asarray(mask, dtype=bool), arr.shape)
    return arr[m]


def horizon_metrics(pred, target, last_in, mask, contact_thr: float = 0.05):
    """Per-horizon metrics. pred/target:(N,Tout,2,21,21) in [0,1]; last_in:(N,2,21,21)
    last input frame (for persistence). Returns dict h-> {mse,mae,skill,force_mae,iou}.
    """
    N, Tout = pred.shape[:2]
    res = {}
    for h in range(Tout):
        p, t = pred[:, h], target[:, h]
        per = np.broadcast_to(last_in, p.shape)  # persistence = last input frame
        mse = float(np.mean(_masked((p - t) ** 2, mask)))
        mae = float(np.mean(np.abs(_masked(p - t, mask))))
        mse_pers = float(np.mean(_masked((per - t) ** 2, mask))) + 1e-12
        skill = 1.0 - mse / mse_pers
        # total force per hand (sum over valid taxels), averaged
        m = mask[None]
        force_p = (p * m).reshape(N, N_HANDS, -1).sum(-1)
        force_t = (t * m).reshape(N, N_HANDS, -1).sum(-1)
        force_mae = float(np.mean(np.abs(force_p - force_t)))
        # contact IoU on binarized maps (valid taxels only)
        pb = (_masked(p, mask) > contact_thr)
        tb = (_masked(t, mask) > contact_thr)
        inter = np.logical_and(pb, tb).sum()
        union = np.logical_or(pb, tb).sum() + 1e-12
        iou = float(inter / union)
        res[h + 1] = dict(mse=mse, mae=mae, skill=skill, force_mae=force_mae, iou=iou)
    return res
