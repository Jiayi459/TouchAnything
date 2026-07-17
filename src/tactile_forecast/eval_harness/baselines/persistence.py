"""Persistence (naive) baseline: y_hat[t+h] = y[t] for all h.

The trivial do-nothing floor. Reads only y[t] -> causal by construction.
"""
from __future__ import annotations

import numpy as np

from .base import Baseline


class Persistence(Baseline):
    name = "persistence"

    def predict(self, hist: np.ndarray, H: int) -> np.ndarray:
        return np.repeat(hist[-1:], H, axis=0)     # (H,6), last observed value
