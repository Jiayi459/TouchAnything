"""CoP masking — the ONE masking rule, imported by every evaluation.

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

from .config import Config


def valid_mask(cfg: Config, target_frames: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Boolean (N, 6): True = include this (frame, channel) in the metric.

    target_frames: (N, 6) the RAW target values at the target time t+h (used to read the
                   force channels). thresholds: per-hand force thresholds (cfg.force_idx order).
    Force channels are always True. Each CoP channel is True iff its hand's force >= thr.
    """
    N = target_frames.shape[0]
    mask = np.ones((N, 6), dtype=bool)
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
