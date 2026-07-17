"""Unit tests for the frozen eval harness, on SYNTHETIC signals with known ground truth.

(a) pure sine: seasonal-naive with the correct period ~ zero error; persistence per-horizon
    MSE matches the analytic value 1 - cos(2*pi*h/T).
(b) AR(2) process: fitted AR recovers the coefficients within tolerance and beats persistence.
(c) masking: frames below the force threshold are excluded from CoP metrics, kept for force.
(d) causality: corrupting the FUTURE (after origin t) never changes a baseline's forecast
    issued at t.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tactile_forecast.eval_harness.config import Config
from src.tactile_forecast.eval_harness.dataset import Norm
from src.tactile_forecast.eval_harness import metrics, masking
from src.tactile_forecast.eval_harness.baselines import (
    Persistence, SeasonalNaive, AR, predict_series)


def make_cfg(horizon, periods=(3, 40), orders=(1, 2, 4), min_history=40, stride=1):
    raw = {
        "target": {"channels": [f"c{i}" for i in range(6)],
                   "force_idx": [0, 3], "cop_idx": [1, 2, 4, 5]},
        "rate": {"fps_raw": 1.0, "downsample": 1, "horizon_s": float(horizon)},
        "mask": {"percentile": 5},
        "baselines": {"ar_orders": list(orders),
                      "seasonal_period_min": periods[0], "seasonal_period_max": periods[1]},
        "eval": {"stride": stride, "min_history": min_history, "seed": 0},
    }
    return Config(raw=raw, path="test", config_hash="test")


def sine_series(T, length, n_ch=6):
    k = np.arange(length)
    return np.stack([np.sin(2 * np.pi * k / T)] * n_ch, axis=1)


# ---------------------------------------------------------------- (a) sine --
def test_seasonal_exact_on_sine():
    T, H = 20, 5
    cfg = make_cfg(H, periods=(3, 40), min_history=40)
    Y = sine_series(T, 20 * T)
    norm = Norm.from_train({0: Y})
    bl = SeasonalNaive(cfg, norm)
    bl.period = T                                   # correct period
    yt, yh = predict_series(bl, {0: Y}, cfg)
    assert np.max(np.abs(yh - yt)) < 1e-9           # seasonal-naive is exact for period T


def test_seasonal_selects_true_period():
    T, H = 20, 5
    cfg = make_cfg(H, periods=(3, 40), min_history=40)
    Y = sine_series(T, 20 * T)
    norm = Norm.from_train({0: Y})
    bl = SeasonalNaive(cfg, norm)
    bl.fit({0: Y})
    bl.select({0: Y}, H)                            # VAL == the sine
    assert bl.period == T


def test_persistence_matches_analytic_sine():
    T, H = 20, 5
    cfg = make_cfg(H, min_history=40)
    Y = sine_series(T, 40 * T)
    bl = Persistence(cfg, Norm.from_train({0: Y}))
    yt, yh = predict_series(bl, {0: Y}, cfg)
    mask = np.ones_like(yt, dtype=bool)
    hz = metrics.masked_horizon_mse(yt, yh, mask)   # (H,6)
    analytic = np.array([1 - np.cos(2 * np.pi * (h + 1) / T) for h in range(H)])
    assert np.allclose(hz[:, 0], analytic, atol=0.02)


# ------------------------------------------------------------- (b) AR(2) --
def test_ar_recovers_coeffs_and_beats_persistence():
    phi1, phi2, H = 0.5, -0.3, 5
    cfg = make_cfg(H, orders=(1, 2, 4), min_history=40)
    rng = np.random.default_rng(0)
    length = 4000
    Z = np.zeros((length, 6))
    for c in range(6):
        z = np.zeros(length)
        for k in range(2, length):
            z[k] = phi1 * z[k - 1] + phi2 * z[k - 2] + 0.1 * rng.standard_normal()
        Z[:, c] = z
    norm = Norm.from_train({0: Z})
    ar = AR(cfg, norm)
    ar.fit({0: Z})
    ar.order = 2
    assert np.allclose(ar.coef[2][:, :2].mean(0), [phi1, phi2], atol=0.05)
    # AR beats persistence on H-step MSE
    yt_a, yh_a = predict_series(ar, {0: Z}, cfg)
    per = Persistence(cfg, norm)
    yt_p, yh_p = predict_series(per, {0: Z}, cfg)
    m = np.ones_like(yt_a, dtype=bool)
    assert metrics.masked_channel_mse(yt_a, yh_a, m).mean() < \
           metrics.masked_channel_mse(yt_p, yh_p, m).mean()


# ------------------------------------------------------------ (c) masking --
def test_cop_masked_below_force_threshold():
    cfg = make_cfg(2)
    N = 100
    tf = np.zeros((N, 6))
    tf[:, 0] = 10.0                      # hand0 force: always high
    tf[:, 3] = 10.0
    tf[:50, 3] = 0.0                     # hand1 force: low for first 50 frames
    thr = np.array([5.0, 5.0])           # per-hand thresholds (force_idx order)
    mask = masking.valid_mask(cfg, tf, thr)
    assert mask[:, 0].all() and mask[:, 3].all()          # force channels never masked
    assert mask[:, 1].all() and mask[:, 2].all()          # hand0 CoP: force high -> kept
    assert (~mask[:50, 4]).all() and (~mask[:50, 5]).all()  # hand1 CoP low-force -> masked
    assert mask[50:, 4].all() and mask[50:, 5].all()      # hand1 CoP high-force -> kept

    # masked CoP error is ignored; force error is not (treat N as the horizon axis)
    ytrue = tf[None].copy()                               # (1,N,6)
    yhat = ytrue.copy()
    yhat[0, :50, 4] += 7.0               # error only on masked (low-force) hand1 CoP-x
    yhat[0, :, 3] += 2.0                 # error on hand1 force (never masked)
    m = mask[None]
    ch_mse = metrics.masked_channel_mse(ytrue, yhat, m)
    assert ch_mse[4] == 0.0              # all ch4 errors were on masked frames
    assert abs(ch_mse[3] - 4.0) < 1e-9   # force error survives (2^2)


# ----------------------------------------------------------- (d) causality --
def test_no_baseline_sees_the_future():
    H = 5
    cfg = make_cfg(H, min_history=40)
    rng = np.random.default_rng(1)
    Y = np.cumsum(rng.standard_normal((300, 6)), axis=0)
    tc = 200                              # corrupt everything at/after tc
    Yc = Y.copy(); Yc[tc:] += 1e4
    norm = Norm.from_train({0: Y})
    for cls in [Persistence, SeasonalNaive, AR]:
        bl = cls(cfg, norm)
        bl.fit({0: Y})
        if isinstance(bl, SeasonalNaive):
            bl.period = 17
        if isinstance(bl, AR):
            bl.order = 2
        for t in (50, 100, 150):          # origins strictly before the corruption
            a = bl.predict(Y[:t + 1], H)
            b = bl.predict(Yc[:t + 1], H)
            assert np.array_equal(a, b), f"{bl.name} changed when future corrupted"
        # sanity: prediction DOES depend on the (whole) past -> test is not vacuous
        Yp = Y.copy(); Yp[:101] += 5.0
        assert not np.array_equal(bl.predict(Y[:101], H), bl.predict(Yp[:101], H))
