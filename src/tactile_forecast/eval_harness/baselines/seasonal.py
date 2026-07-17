"""Seasonal-naive baseline: y_hat[t+h] = y at the same phase, one or more whole periods
back so the referenced sample is always observed (<= t).

    idx = (t + h) - ceil(h / m) * m      (guaranteed <= t, so strictly causal)

with period m (in frames). CANDIDATE periods come from the config range and are ranked on
TRAIN by autocorrelation strength (estimation on TRAIN only); the FINAL period is chosen on
VAL by normalized H-step MSE (selection on VAL only). TEST is never consulted here.
"""
from __future__ import annotations

import numpy as np

from .base import Baseline, predict_series


class SeasonalNaive(Baseline):
    name = "seasonal"

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        pmin = cfg.raw["baselines"]["seasonal_period_min"]
        pmax = cfg.raw["baselines"]["seasonal_period_max"]
        self.candidates = list(range(pmin, pmax + 1))
        self.period = self.candidates[0]

    # -- estimation on TRAIN: rank candidate periods by mean causal autocorrelation --
    def fit(self, train: dict[int, np.ndarray]) -> None:
        z = np.concatenate([self.norm.z(Y) for Y in train.values()], axis=0)  # (sumT,6)
        z = z - z.mean(0)
        var = (z * z).mean(0) + 1e-12
        score = {}
        for m in self.candidates:
            if m >= len(z):
                continue
            ac = (z[m:] * z[:-m]).mean(0) / var        # per-channel autocorr at lag m
            score[m] = float(ac.mean())                # average across channels
        # keep candidates ordered by TRAIN autocorr (best first); selection still on VAL
        self.candidates = sorted(score, key=lambda m: -score[m]) or self.candidates
        self.period = self.candidates[0]

    # -- selection on VAL: pick the period with lowest normalized H-step MSE --
    def select(self, val: dict[int, np.ndarray], H: int) -> None:
        best, best_err = self.period, np.inf
        for m in self.candidates:
            self.period = m
            yt, yh = predict_series(self, val, self.cfg)
            if len(yt) == 0:
                continue
            err = float((((yh - yt) / self.norm.std) ** 2).mean())   # normalized MSE
            if err < best_err:
                best, best_err = m, err
        self.period = best

    def predict(self, hist: np.ndarray, H: int) -> np.ndarray:
        t = len(hist) - 1
        m = self.period
        out = np.empty((H, 6))
        for h in range(1, H + 1):
            k = -(-h // m)                       # ceil(h/m)
            idx = (t + h) - k * m
            out[h - 1] = hist[idx] if idx >= 0 else hist[-1]   # causal: idx <= t always
        return out
