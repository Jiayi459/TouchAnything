"""CoP masking — fork of src/actionsense/eval_harness/masking.py.

FORKED, NOT SHARED (per user decision 2026-08-10: src/actionsense/ is never edited, even
for changes that are no-ops on ActionSense's own 6-channel config). The only change from
the original: the channel count is read from the array shape instead of hardcoded 6, so
this works for OpenTouch's 3-channel target. Everything else — the masking RULE itself —
is identical. If you find a bug here, check whether the same bug exists in the ActionSense
original; fixes do not propagate automatically between the two.

CoP (pressure-weighted centroid) is numerically undefined when contact force is near zero
(the centroid of an all-zero map is arbitrary). So a CoP TARGET frame is EXCLUDED from CoP
metrics iff that hand's RAW total force at that frame is below the TRAIN per-hand threshold
(see dataset.force_thresholds). Force channels are NEVER masked.

The mask keys off the RAW total force at the TARGET frame (the physical contact at the time
the predicted value refers to), regardless of what quantity the model predicts. This is a
pure element-wise comparison — no filtering, trivially causal at the metric level.
"""
from __future__ import annotations

import numpy as np

from src.actionsense.eval_harness.config import Config


def valid_mask(cfg: Config, target_frames: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Boolean (N, C): True = include this (frame, channel) in the metric.

    target_frames: (N, C) the RAW target values at the target time t+h (used to read the
                   force channels). thresholds: per-hand force thresholds (cfg.force_idx order).
    Force channels are always True. Each CoP channel is True iff its hand's force >= thr.
    """
    N, C = target_frames.shape
    mask = np.ones((N, C), dtype=bool)
    # map each CoP channel to its hand's force channel + threshold
    for hand, fi in enumerate(cfg.force_idx):
        thr = thresholds[hand]
        contact = target_frames[:, fi] >= thr                 # (N,)
        # CoP channels belonging to this hand: those cop_idx between this force_idx and the next
        force_sorted = sorted(cfg.force_idx)
        lo = fi
        hi = force_sorted[force_sorted.index(fi) + 1] if force_sorted.index(fi) + 1 < len(force_sorted) else 10**9
        for ci in cfg.cop_idx:
            if lo < ci < hi:
                mask[:, ci] = contact
    return mask
