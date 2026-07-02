"""Training-free tactile predictability metrics (numpy-only, sensor-agnostic).

Shared by the per-dataset predictability probes so every dataset is scored with
IDENTICAL math (EgoTouch 21x21 two-hand, OpenTouch 16x16 one-hand, ...). A "sequence"
is any array (T, C, H, W): T frames, C channels/hands, H x W taxel grid.

Metrics (see docs/ACTION_CATEGORIES.md):
  RAW HARDNESS   persistence_nMSE[h] = MSE(y[t+h], y[t]) / Var(y)      (lower = easier)
  STRUCTURE      periodicity = max total-force autocorr, lag in [PERIOD_LO, PERIOD_HI]
  CONTACT        contact_migration = 1 - IoU(active mask t, t+h)       (lower = stable)
  COMPOSITE      predictability_index = z(-persH_mid) + z(periodicity) + z(-migration)

Metrics are scale-invariant (nMSE normalized by variance; periodicity is a correlation;
migration uses a fractional-of-max threshold), so raw sensor units need no normalization.
Lag windows assume ~30 fps (EgoTouch and OpenTouch both record at 30 Hz).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

FPS = 30
HORIZONS = (1, 5, 15, 30)          # frames: 33 / 167 / 500 / 1000 ms
H_MID = 15                          # horizon used for the composite index
PERIOD_LO, PERIOD_HI = 10, 45      # lag window (0.33-1.5 s) for the periodicity peak
ACTIVE_FRAC = 0.05                 # taxel "in contact" if > 5% of the sequence max

METRIC_KEYS = ("pers_nmse_h1", "pers_nmse_h15", "pers_nmse_h30",
               "periodicity", "contact_migration")


def seq_metrics(s: np.ndarray) -> dict:
    """Scalar predictability metrics for one (T, C, H, W) sequence."""
    s = np.asarray(s, dtype=np.float32)
    T = s.shape[0]
    var = float(np.var(s)) + 1e-8
    out: dict = {}
    for h in HORIZONS:
        if T <= h + 1:
            continue
        a, b = s[:-h], s[h:]
        out[f"pers_nmse_h{h}"] = float(np.mean((b - a) ** 2)) / var
    # periodicity: max autocorr of total force at lag in [PERIOD_LO, PERIOD_HI]
    f = s.reshape(T, -1).sum(1)
    if f.std() > 1e-6:
        f0 = f - f.mean()
        best = 0.0
        for lag in range(PERIOD_LO, min(PERIOD_HI, T - 1) + 1):
            a, b = f0[:-lag], f0[lag:]
            denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
            best = max(best, float((a * b).sum() / denom))
        out["periodicity"] = best
    # contact migration at H_MID: 1 - IoU of active-taxel masks
    if T > H_MID:
        thr = ACTIVE_FRAC * float(s.max() + 1e-8)
        m0, m1 = s[:-H_MID] > thr, s[H_MID:] > thr
        inter = np.logical_and(m0, m1).sum(axis=(1, 2, 3))
        union = np.logical_or(m0, m1).sum(axis=(1, 2, 3)).astype(np.float64)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        out["contact_migration"] = float(1.0 - iou.mean())
    return out


def aggregate(groups: dict) -> dict:
    """groups: label -> list[seq_metrics dict]  ->  label -> {n, mean metrics}."""
    rows = {}
    for g, lst in groups.items():
        row = {"n": len(lst)}
        for k in METRIC_KEYS:
            vals = [d[k] for d in lst if k in d and np.isfinite(d[k])]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        rows[g] = row
    return rows


def add_predictability_index(rows: dict) -> dict:
    """PI = z(-pers_nmse_h{H_MID}) + z(periodicity) + z(-contact_migration)."""
    def zcol(key, sign):
        vals = np.array([sign * rows[g][key] for g in rows], float)
        mu, sd = np.nanmean(vals), np.nanstd(vals) + 1e-8
        return {g: (sign * rows[g][key] - mu) / sd for g in rows}
    zp = zcol(f"pers_nmse_h{H_MID}", -1.0)
    zper = zcol("periodicity", 1.0)
    zm = zcol("contact_migration", -1.0)
    for g in rows:
        rows[g]["pi"] = zp[g] + zper[g] + zm[g]
    return rows


def new_group_dict():
    """Convenience: a defaultdict(list) for accumulating per-label seq_metrics."""
    return defaultdict(list)
