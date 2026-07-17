"""Metrics — the ONE scoring definition, imported by every evaluation.

All predictions are shaped (N, H, 6): N forecast origins, H horizon steps (indexed by
TARGET time t+h, h = 1..H), 6 channels. `mask` is the same shape (from masking.valid_mask,
broadcast over horizon); masked-out (frame, channel) entries are excluded from that
channel's metric. Errors are Mean-Squared-Error unless noted; skill is vs a reference
(persistence) baseline: skill = 1 - MSE_model / MSE_ref (0 = tie, 1 = perfect, <0 = worse).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def masked_channel_mse(ytrue: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,H,6) -> (6,): MSE per channel, aggregated over all valid origins & horizons."""
    e = (yhat - ytrue) ** 2
    num = (e * mask).reshape(-1, 6).sum(0)
    den = mask.reshape(-1, 6).sum(0)
    return num / np.maximum(den, 1.0)


def masked_horizon_mse(ytrue: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,H,6) -> (H,6): MSE per (horizon step, channel)."""
    e = (yhat - ytrue) ** 2
    num = (e * mask).sum(0)          # (H,6)
    den = mask.sum(0)                # (H,6)
    return num / np.maximum(den, 1.0)


def masked_channel_mae(ytrue: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    e = np.abs(yhat - ytrue)
    num = (e * mask).reshape(-1, 6).sum(0)
    den = mask.reshape(-1, 6).sum(0)
    return num / np.maximum(den, 1.0)


def skill(mse_model: np.ndarray, mse_ref: np.ndarray) -> np.ndarray:
    """1 - MSE_model / MSE_ref, elementwise (guarded)."""
    return 1.0 - mse_model / (mse_ref + EPS)


def normalized_rmse(mse_per_channel: np.ndarray, std: np.ndarray) -> float:
    """Headline scalar: RMSE per channel divided by that channel's TRAIN std, averaged.
    Puts force (large units) and CoP (small units) on a comparable scale."""
    return float(np.mean(np.sqrt(mse_per_channel) / (std + EPS)))
