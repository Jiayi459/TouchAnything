"""Autoregressive AR(p) baseline, per channel, fit by ordinary least squares.

    z[k] ~= b + sum_{i=1..p} phi_i * z[k-i]

where z is the GLOBALLY normalized signal (dataset.Norm, TRAIN-derived). Coefficients are
fit on TRAIN only (for every candidate order); the order p is selected on VAL by normalized
H-step MSE. Forecasting is the standard recursion seeded by the last p observed values and
then rolled forward on its own predictions — reads only `hist` (times <= t), so causal.
numpy only (no statsmodels): OLS via lstsq.
"""
from __future__ import annotations

import numpy as np

from .base import Baseline, predict_series


def _design(z_list: list[np.ndarray], p: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack lagged rows for one channel across recordings. z_list: list of (T,) arrays."""
    X, y = [], []
    for z in z_list:
        if len(z) <= p:
            continue
        for k in range(p, len(z)):
            X.append(np.concatenate([z[k - p:k][::-1], [1.0]]))   # [z[k-1],..,z[k-p],1]
            y.append(z[k])
    return np.asarray(X), np.asarray(y)


class AR(Baseline):
    name = "ar"

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.orders = list(cfg.raw["baselines"]["ar_orders"])
        self.order = self.orders[0]
        self.coef: dict[int, np.ndarray] = {}     # order -> (6, p+1): [phi_1..phi_p, b]

    def fit(self, train: dict[int, np.ndarray]) -> None:
        zt = [self.norm.z(Y) for Y in train.values()]
        for p in self.orders:
            C = np.zeros((6, p + 1))
            for c in range(6):
                X, y = _design([z[:, c] for z in zt], p)
                if len(X):
                    C[c], *_ = np.linalg.lstsq(X, y, rcond=None)
            self.coef[p] = C

    def select(self, val: dict[int, np.ndarray], H: int) -> None:
        best, best_err = self.order, np.inf
        for p in self.orders:
            self.order = p
            yt, yh = predict_series(self, val, self.cfg)
            if len(yt) == 0:
                continue
            err = float((((yh - yt) / self.norm.std) ** 2).mean())
            if err < best_err:
                best, best_err = p, err
        self.order = best

    def predict(self, hist: np.ndarray, H: int) -> np.ndarray:
        p = self.order
        C = self.coef[p]                                  # (6, p+1)
        z = self.norm.z(hist)                             # (t+1, 6)
        phi, b = C[:, :p], C[:, p]                        # (6,p), (6,)
        buf = list(z[-p:]) if len(z) >= p else list(np.zeros((p - len(z), 6))) + list(z)
        out = np.empty((H, 6))
        for h in range(H):
            recent = np.stack(buf[-p:])[::-1]             # (p,6): lag1..lagp
            nxt = b + (phi * recent.T).sum(1)             # (6,)
            out[h] = nxt
            buf.append(nxt)
        return self.norm.unz(out)                         # back to raw units
