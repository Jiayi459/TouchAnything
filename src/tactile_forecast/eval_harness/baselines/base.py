"""Baseline contract + the causal rolling-origin batcher.

Contract (all baselines): given `hist` = observations at times [0..t] (shape (t+1, 6)),
`predict(hist, H)` returns (H, 6), the forecast for target times t+1..t+H. It MUST read only
`hist` — never any value at time > t. `predict_series` below builds the full evaluation
tensor by calling predict once per origin, so causality is guaranteed structurally: the
batcher only ever hands a baseline the past slice Y[:t+1].
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import Config
from ..dataset import Norm


class Baseline:
    name = "base"

    def __init__(self, cfg: Config, norm: Norm):
        self.cfg = cfg
        self.norm = norm

    def fit(self, train: dict[int, np.ndarray]) -> None:
        """Estimate global parameters from TRAIN only. Default: nothing to fit."""

    def select(self, val: dict[int, np.ndarray], H: int) -> None:
        """Select hyperparameters on VAL only. Default: nothing to select."""

    def predict(self, hist: np.ndarray, H: int) -> np.ndarray:  # (t+1,6) -> (H,6)
        raise NotImplementedError


def origins(T: int, cfg: Config) -> np.ndarray:
    """Valid forecast origins t: enough history behind, full horizon ahead."""
    lo = cfg.raw["eval"]["min_history"]
    stride = cfg.raw["eval"]["stride"]
    hi = T - cfg.horizon                       # need t+H <= T-1  => t <= T-1-H
    return np.arange(lo, hi, stride)


def predict_series(bl: Baseline, group: dict[int, np.ndarray], cfg: Config
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin evaluation over a set of recordings.

    Returns (ytrue, yhat), each (N_total, H, 6): for every valid origin t in every
    recording, the true future Y[t+1..t+H] and the baseline's causal forecast.
    """
    H = cfg.horizon
    yts, yhs = [], []
    for _, Y in sorted(group.items()):
        for t in origins(len(Y), cfg):
            yts.append(Y[t + 1:t + 1 + H])                 # (H,6) target-time indexed
            yhs.append(bl.predict(Y[:t + 1], H))           # causal: only past passed in
    if not yts:
        return np.zeros((0, H, 6)), np.zeros((0, H, 6))
    return np.stack(yts), np.stack(yhs)
