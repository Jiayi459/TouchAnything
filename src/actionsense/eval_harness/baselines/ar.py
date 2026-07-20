"""Linear AR baseline — per channel, fit PER GROUP (activity x object) on TRAIN.

    z[k] ~= b + sum_{i=1..p} phi_i * z[k-i]      (z = global TRAIN-normalized signal)

Coefficients come from statsmodels AutoReg (trend='c') fit on the group's TRAIN series (numpy
OLS fallback if statsmodels is unavailable). The order p is selected PER GROUP on VAL from the
config candidate set by iterated H-step normalized MSE. Multi-step forecasts are the standard
iterated recursion seeded by the last p observed values -> reads only `hist` (times <= t), causal.

Pooling note: a group's TRAIN recordings are concatenated for the AutoReg fit; with long clips
the few cross-recording lag rows are negligible.
"""
from __future__ import annotations

import numpy as np

from .base import Baseline, by_group, predict_series

try:
    from statsmodels.tsa.ar_model import AutoReg
    HAVE_SM = True
except Exception:                          # pragma: no cover
    HAVE_SM = False


def _fit_channel(series: np.ndarray, p: int) -> np.ndarray:
    """Fit AR(p) to a 1-D series -> [phi_1..phi_p, b]. statsmodels AutoReg, numpy OLS fallback."""
    if HAVE_SM and len(series) > p + 1:
        res = AutoReg(series, lags=p, trend="c", old_names=False).fit()
        pr = np.asarray(res.params)        # [const, L1, .., Lp]
        return np.concatenate([pr[1:1 + p], pr[:1]])
    # numpy OLS fallback: rows [z[k-1..k-p], 1] -> z[k]
    X, y = [], []
    for k in range(p, len(series)):
        X.append(np.concatenate([series[k - p:k][::-1], [1.0]]))
        y.append(series[k])
    if not X:
        return np.zeros(p + 1)
    coef, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
    return coef                            # [phi_1..phi_p, b]


class AR(Baseline):
    name = "ar"

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.orders = list(cfg.raw["baselines"]["ar_orders"])
        # coef[group][order] -> (6, p+1) = [phi_1..phi_p, b] per channel
        self.coef: dict[str, dict[int, np.ndarray]] = {}
        self.order: dict[str, int] = {}

    def fit(self, train: dict[int, np.ndarray], groups: dict[int, str]) -> None:
        for g, recs in by_group(train, groups).items():
            z = np.concatenate([self.norm.z(Y) for Y in recs.values()], axis=0)   # (sumT,6)
            self.coef[g] = {}
            for p in self.orders:
                C = np.zeros((6, p + 1))
                for c in range(6):
                    C[c] = _fit_channel(z[:, c], p)
                self.coef[g][p] = C
            self.order[g] = self.orders[0]

    def select(self, val: dict[int, np.ndarray], groups: dict[int, str], H: int) -> None:
        vg = by_group(val, groups)
        global_best = self._best_order(val, groups, H)
        for g in self.coef:
            self.order[g] = self._best_order(vg.get(g, {}),
                                             {i: g for i in vg.get(g, {})}, H, default=global_best)

    def _best_order(self, data, groups, H, default=None):
        """Order minimizing iterated H-step normalized MSE on `data`. Empty data -> default."""
        if not data:
            return default if default is not None else self.orders[0]
        present = set(groups.values())
        best, best_err = default if default is not None else self.orders[0], np.inf
        for p in self.orders:
            for g in present:                         # temporarily set this order, measure
                self.order[g] = p
            yt, yh = predict_series(self, data, groups, self.cfg)
            err = float((((yh - yt) / self.norm.std) ** 2).mean())
            if err < best_err:
                best, best_err = p, err
        return best

    def predict(self, hist: np.ndarray, H: int, group: str) -> np.ndarray:
        p = self.order[group]
        C = self.coef[group][p]                       # (6, p+1)
        z = self.norm.z(hist)
        phi, b = C[:, :p], C[:, p]
        buf = list(z[-p:]) if len(z) >= p else list(np.zeros((p - len(z), 6))) + list(z)
        out = np.empty((H, 6))
        for h in range(H):
            recent = np.stack(buf[-p:])[::-1]          # (p,6): lag1..lagp
            nxt = b + (phi * recent.T).sum(1)
            out[h] = nxt
            buf.append(nxt)
        return self.norm.unz(out)
