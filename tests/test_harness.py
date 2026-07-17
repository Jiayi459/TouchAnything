"""Unit tests for the frozen eval harness, on SYNTHETIC signals with known ground truth.

(a) pure sine: seasonal-naive with the correct period ~ zero error; the period is RECOVERED from
    the TRAIN autocorrelation; persistence per-horizon MSE matches analytic 1 - cos(2*pi*h/T).
(b) AR(2) process: fitted AR (statsmodels AutoReg / numpy fallback) recovers the coefficients
    within tolerance and beats persistence.
(c) masking: frames below the force threshold are excluded from CoP metrics, kept for force.
(d) causality: corrupting the FUTURE (after origin t) never changes a baseline's forecast at t.
(e) seasonal fallback: a signal with no cycle falls back to persistence (period = None).
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

G = {0: "g0"}          # single synthetic group


def make_cfg(horizon, pmin_s=0.2, pmax_s=4.0, orders=(2, 5), min_history=45, stride=1):
    raw = {
        "target": {"channels": [f"c{i}" for i in range(6)],
                   "force_idx": [0, 3], "cop_idx": [1, 2, 4, 5]},
        "rate": {"fps_raw": 1.0, "downsample": 1, "horizon_s": float(horizon)},
        "mask": {"percentile": 5},
        "baselines": {"fit_scope": "group", "ar_orders": list(orders),
                      "seasonal_period_min_s": pmin_s, "seasonal_period_max_s": pmax_s,
                      "seasonal_min_autocorr": 0.1},
        "eval": {"stride": stride, "min_history": min_history, "seed": 0},
    }
    return Config(raw=raw, path="test", config_hash="test")


def sine_series(T, length, n_ch=6):
    k = np.arange(length)
    return np.stack([np.sin(2 * np.pi * k / T)] * n_ch, axis=1)


# ---------------------------------------------------------------- (a) sine --
def test_seasonal_exact_on_sine():
    T, H = 20, 5
    cfg = make_cfg(H, pmax_s=40.0, min_history=45)
    Y = sine_series(T, 20 * T)
    bl = SeasonalNaive(cfg, Norm.from_train({0: Y}))
    bl.periods = {"g0": T}
    yt, yh = predict_series(bl, {0: Y}, G, cfg)
    assert np.max(np.abs(yh - yt)) < 1e-9


def test_seasonal_recovers_period_from_train_autocorr():
    T, H = 20, 5
    cfg = make_cfg(H, pmax_s=40.0, min_history=45)
    Y = sine_series(T, 20 * T)
    bl = SeasonalNaive(cfg, Norm.from_train({0: Y}))
    bl.fit({0: Y}, G)                      # estimate period on TRAIN autocorrelation
    assert bl.periods["g0"] == T


def test_persistence_matches_analytic_sine():
    T, H = 20, 5
    cfg = make_cfg(H, min_history=45)
    Y = sine_series(T, 40 * T)
    bl = Persistence(cfg, Norm.from_train({0: Y}))
    yt, yh = predict_series(bl, {0: Y}, G, cfg)
    hz = metrics.masked_horizon_mse(yt, yh, np.ones_like(yt, dtype=bool))
    analytic = np.array([1 - np.cos(2 * np.pi * (h + 1) / T) for h in range(H)])
    assert np.allclose(hz[:, 0], analytic, atol=0.02)


# ------------------------------------------------------------- (b) AR(2) --
def test_ar_recovers_coeffs_and_beats_persistence():
    phi1, phi2, H = 0.5, -0.3, 5
    cfg = make_cfg(H, orders=(2, 5), min_history=45)
    rng = np.random.default_rng(0)
    n = 4000
    Z = np.zeros((n, 6))
    for c in range(6):
        z = np.zeros(n)
        for k in range(2, n):
            z[k] = phi1 * z[k - 1] + phi2 * z[k - 2] + 0.1 * rng.standard_normal()
        Z[:, c] = z
    norm = Norm.from_train({0: Z})
    ar = AR(cfg, norm)
    ar.fit({0: Z}, G)
    ar.order["g0"] = 2
    assert np.allclose(ar.coef["g0"][2][:, :2].mean(0), [phi1, phi2], atol=0.05)
    yt_a, yh_a = predict_series(ar, {0: Z}, G, cfg)
    per = Persistence(cfg, norm)
    yt_p, yh_p = predict_series(per, {0: Z}, G, cfg)
    m = np.ones_like(yt_a, dtype=bool)
    assert metrics.masked_channel_mse(yt_a, yh_a, m).mean() < \
           metrics.masked_channel_mse(yt_p, yh_p, m).mean()


# ------------------------------------------------------------ (c) masking --
def test_cop_masked_below_force_threshold():
    cfg = make_cfg(2)
    N = 100
    tf = np.zeros((N, 6))
    tf[:, 0] = 10.0; tf[:, 3] = 10.0
    tf[:50, 3] = 0.0                     # hand1 force low for first 50 frames
    thr = np.array([5.0, 5.0])
    mask = masking.valid_mask(cfg, tf, thr)
    assert mask[:, 0].all() and mask[:, 3].all()          # force never masked
    assert mask[:, 1].all() and mask[:, 2].all()          # hand0 CoP kept
    assert (~mask[:50, 4]).all() and (~mask[:50, 5]).all()  # hand1 CoP masked when low
    assert mask[50:, 4].all() and mask[50:, 5].all()

    ytrue = tf[None].copy(); yhat = ytrue.copy()
    yhat[0, :50, 4] += 7.0               # error only on masked frames
    yhat[0, :, 3] += 2.0                 # error on (never-masked) force
    ch_mse = metrics.masked_channel_mse(ytrue, yhat, mask[None])
    assert ch_mse[4] == 0.0
    assert abs(ch_mse[3] - 4.0) < 1e-9


# ----------------------------------------------------------- (d) causality --
def test_no_baseline_sees_the_future():
    H = 5
    cfg = make_cfg(H, min_history=45)
    rng = np.random.default_rng(1)
    Y = np.cumsum(rng.standard_normal((300, 6)), axis=0)
    tc = 200
    Yc = Y.copy(); Yc[tc:] += 1e4
    norm = Norm.from_train({0: Y})
    for cls in [Persistence, SeasonalNaive, AR]:
        bl = cls(cfg, norm)
        bl.fit({0: Y}, G)
        if isinstance(bl, SeasonalNaive):
            bl.periods = {"g0": 17}
        if isinstance(bl, AR):
            bl.order["g0"] = 2
        for t in (50, 100, 150):
            a = bl.predict(Y[:t + 1], H, "g0")
            b = bl.predict(Yc[:t + 1], H, "g0")
            assert np.array_equal(a, b), f"{bl.name} changed when future corrupted"
        Yp = Y.copy(); Yp[:101] += 5.0
        assert not np.array_equal(bl.predict(Y[:101], H, "g0"), bl.predict(Yp[:101], H, "g0"))


# ------------------------------------------------------- (e) seasonal fallback --
def test_seasonal_fallback_when_no_cycle():
    H = 5
    cfg = make_cfg(H, min_history=45)
    rng = np.random.default_rng(2)
    Y = np.cumsum(rng.standard_normal((600, 6)), axis=0)   # random walk: no fixed cycle
    bl = SeasonalNaive(cfg, Norm.from_train({0: Y}))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl.fit({0: Y}, G)
    assert bl.periods["g0"] is None                        # fell back
    per = Persistence(cfg, Norm.from_train({0: Y}))
    assert np.array_equal(bl.predict(Y[:120], H, "g0"), per.predict(Y[:120], H, "g0"))
