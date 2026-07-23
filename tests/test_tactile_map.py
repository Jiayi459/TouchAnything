"""Unit tests for the tactile-map -> F/CoP forecaster pipeline (synthetic; no training)."""
import glob
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.actionsense.eval_harness.config import Config
from src.actionsense.eval_harness.baselines.base import origins
from src.actionsense.tactile_map import data as D
from src.actionsense.tactile_map.models import build_model


def make_cfg(horizon=3, min_history=5, downsample=1, stride=1, root="data/actionsense_states"):
    raw = {
        "target": {"channels": [f"c{i}" for i in range(6)], "force_idx": [0, 3], "cop_idx": [1, 2, 4, 5]},
        "rate": {"fps_raw": 1.0, "downsample": downsample, "horizon_s": float(horizon)},
        "eval": {"stride": stride, "min_history": min_history, "seed": 0},
        "paths": {"states_root": root},
    }
    return Config(raw=raw, path="test", config_hash="test")


def rand_maps(T, n=1, seed=0):
    rng = np.random.default_rng(seed)
    return {i: rng.random((T, 2, 32, 32)).astype(np.float32) for i in range(n)}


# --- (i) causality: corrupting the FUTURE never changes a window issued at origin t ---
def test_window_is_causal():
    cfg = make_cfg(horizon=3, min_history=5); t_in = 4
    M = rand_maps(40, 1)[0]
    tc = 25
    Mc = M.copy(); Mc[tc:] += 5.0                       # corrupt the future
    ds, dsc = D.MapWindows({0: M}, {0: np.zeros((40, 6), np.float32)}, cfg, t_in), \
              D.MapWindows({0: Mc}, {0: np.zeros((40, 6), np.float32)}, cfg, t_in)
    for t in [i for i in origins(40, cfg) if i < tc]:
        assert np.array_equal(ds._window(0, t), dsc._window(0, t)), f"window at t={t} saw the future"


# --- (ii) export alignment: one window per harness origin, correct shape ---
def test_recording_windows_align_to_origins():
    cfg = make_cfg(horizon=3, min_history=5); t_in = 4
    M = rand_maps(50, 1)[0]
    X, ors = D.recording_windows(M, cfg, t_in)
    assert len(ors) == len(origins(len(M), cfg))
    assert X.shape == (len(ors), t_in, 2, 32, 32)


# --- (iii) left-pad early origins with zeros; real tail preserved ---
def test_left_pad_early_origin():
    cfg = make_cfg(horizon=3, min_history=2); t_in = 10   # min_history < t_in -> early origins pad
    M = rand_maps(30, 1)[0]
    ds = D.MapWindows({0: M}, {0: np.zeros((30, 6), np.float32)}, cfg, t_in)
    t = 4                                               # only 5 real frames (0..4) < t_in=10
    w = ds._window(0, t)
    assert w.shape[0] == t_in
    assert np.all(w[:t_in - (t + 1)] == 0.0)            # leading pad is zeros
    assert np.array_equal(w[t_in - (t + 1):], M[: t + 1])   # tail is the real history


# --- (iv) log1p compression: zero-preserving, monotone, sub-linear ---
def test_compress_properties():
    assert D.compress(np.array([0.0]), 10.0)[0] == 0.0
    x = np.array([0.1, 1.0, 10.0, 100.0])
    c = D.compress(x, 10.0)
    assert np.all(np.diff(c) > 0)                       # monotone increasing
    assert c[3] / c[2] < x[3] / x[2]                    # large values compressed


# --- (v) baseline uses only the first N frames (causal) ---
def test_baseline_first_n_only(tmp_path):
    cfg = make_cfg(downsample=1, root=str(tmp_path))
    clip = np.ones((20, 2, 32, 32), np.float32) * 3.0
    clip[10:] += 100.0                                  # a big future spike
    np.save(os.path.join(str(tmp_path), "clip_0.npy"), clip)
    m = D.load_map(cfg, 0, baseline_frames=5)           # base = mean of first 5 (=3) -> early frames ~0
    assert np.allclose(m[0], 0.0)                       # baseline removed the resting 3.0
    assert np.allclose(m[10], 100.0)                    # the later spike survives (base is past-only)


# --- (vi) model output shape (both encoders) ---
@pytest.mark.parametrize("enc", ["flatten", "cnn"])
def test_model_shape(enc):
    m = build_model(enc, horizon=10, d=32, hidden=32)
    mu, lv = m(torch.randn(2, 7, 2, 32, 32))          # probabilistic: (mean, log-variance)
    assert mu.shape == lv.shape == (2, 10, 6)
    assert lv.min() >= -6 - 1e-4 and lv.max() <= 4 + 1e-4   # clamped


def test_aggregate_model_and_windows():
    m = build_model("aggregate", horizon=10, d=32, hidden=32)
    mu, lv = m(torch.randn(2, 7, 6))                  # aggregate input: (B, t_in, 6)
    assert mu.shape == lv.shape == (2, 10, 6)
    cfg = make_cfg(horizon=3, min_history=5); t_in = 4
    S = np.cumsum(np.random.default_rng(0).standard_normal((40, 6)), 0).astype(np.float32)
    ds = D.AggWindows({0: S}, cfg, t_in)
    x, y = ds[3]; i, t = ds.index[3]
    assert x.shape == (t_in, 6) and y.shape == (3, 6)
    assert np.allclose(y.numpy(), S[t + 1:t + 4] - S[t], atol=1e-5)   # residual-over-persistence


# --- (viii) residual-over-persistence target = future - last observed value ---
def test_residual_target():
    cfg = make_cfg(horizon=3, min_history=5); t_in = 4
    M = rand_maps(40, 1)[0]
    Y = np.cumsum(np.random.default_rng(0).standard_normal((40, 6)), 0).astype(np.float32)
    ds = D.MapWindows({0: M}, {0: Y}, cfg, t_in)
    k = 3; i, t = ds.index[k]
    _, y = ds[k]
    assert np.allclose(y.numpy(), Y[t + 1:t + 1 + cfg.horizon] - Y[t], atol=1e-5)


# --- (vii) integration on a real cached map, if any exist locally ---
def test_real_map_loads_if_available():
    files = glob.glob("data/actionsense_states/clip_*.npy")
    if not files:
        pytest.skip("no cached maps locally")
    idx = int(os.path.basename(files[0]).split("_")[1].split(".")[0])
    cfg = make_cfg()  # downsample from harness is 3; here just check load works with real ds
    from src.actionsense.eval_harness.config import load_config
    hcfg = load_config()
    m = D.load_map(hcfg, idx, baseline_frames=10)
    assert m.ndim == 4 and m.shape[1:] == (2, 32, 32) and np.all(m >= 0)
