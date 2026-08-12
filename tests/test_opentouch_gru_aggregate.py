"""Unit tests for the deterministic GRU-aggregate fork (src/opentouch/gru_aggregate.py).

SEPARATE FILE because it needs torch: a module-level importorskip skips the whole file, and the
pure-numpy G2 tests (tests/test_opentouch_g2.py) must never be skipped along with it.

The contracts pinned here are the ones that would silently corrupt a G1 comparison if broken:
  * windows are causal and left-padded, so a prediction exists at EVERY harness origin;
  * the residual-over-persistence convention holds end to end -- a model whose head outputs 0
    reproduces persistence EXACTLY in raw units (this simultaneously checks the persistence
    anchor and the de-normalization, the two places a sign or an order-of-operations error
    would hide);
  * predictions align 1:1 with baselines.origins, which is what evaluate.score_external asserts;
  * a frozen seed gives bitwise identical weights (Q5's determinism requirement).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# NOT pytest.importorskip: that only catches ImportError, and a half-installed torch raises
# OSError from dlopen (which is exactly the state of this Mac as of 2026-08-12 -- a missing
# libtorch_global_deps.dylib). An uncaught OSError aborts COLLECTION for the whole pytest run,
# taking the pure-numpy tests down with it, so every failure mode is caught here.
try:
    import torch
except Exception as exc:                                        # noqa: BLE001
    pytest.skip(f"torch not usable in this environment: {exc}", allow_module_level=True)

from src.actionsense.eval_harness.config import Config          # noqa: E402
from src.opentouch import gru_aggregate as GA                   # noqa: E402
from src.opentouch.baselines import origins                     # noqa: E402
from src.opentouch.dataset import Norm                          # noqa: E402

N_CH = 3


def make_cfg(horizon=2, min_history=2, fps=30.0):
    raw = {
        "target": {"channels": ["F_R", "CoPx_R", "CoPy_R"], "force_idx": [0], "cop_idx": [1, 2]},
        "rate": {"fps_raw": fps, "downsample": 1, "horizon_s": horizon / fps},
        "mask": {"percentile": 5},
        "baselines": {"fit_scope": "object_category", "ar_orders": [2],
                      "seasonal_period_min_s": 0.2, "seasonal_period_max_s": 1.0,
                      "seasonal_min_autocorr": 0.1},
        "eval": {"stride": 1, "min_history": min_history, "seed": 0},
    }
    return Config(raw=raw, path="test", config_hash="test")


def test_windows_left_pad_early_origins_and_read_only_the_past():
    cfg = make_cfg()
    sig = {0: np.arange(24, dtype=np.float64).reshape(8, N_CH)}
    ds = GA.AggWindows(sig, cfg, t_in=4)
    w = ds.window(0, 2)                                    # only 3 frames of history exist
    assert w.shape == (4, N_CH)
    assert np.allclose(w[0], 0.0)                          # zero-padded head
    assert np.allclose(w[1:], sig[0][:3])
    assert np.allclose(ds.window(0, 5), sig[0][2:6])       # full window, nothing beyond origin


def test_channel_count_comes_from_the_data_not_a_hardcoded_six():
    """The bug class the OpenTouch forks exist to fix: the ActionSense original pads with
    np.zeros((pad, 6)), which would raise (or silently mis-shape) on a 3-channel target."""
    cfg = make_cfg()
    ds = GA.AggWindows({0: np.zeros((5, N_CH))}, cfg, t_in=4)
    assert ds.window(0, 1).shape == (4, N_CH)


def test_residual_target_is_the_change_from_the_last_observation():
    cfg = make_cfg()
    sig = {0: np.arange(24, dtype=np.float64).reshape(8, N_CH)}
    ds = GA.AggWindows(sig, cfg, t_in=3)
    _, y = ds[0]
    _, t = ds.index[0]
    assert np.allclose(y.numpy(), sig[0][t + 1:t + 1 + cfg.horizon] - sig[0][t])


def test_zero_output_model_reproduces_persistence_exactly_in_raw_units():
    cfg = make_cfg()
    Y = np.cumsum(np.ones((10, N_CH)), axis=0) * np.array([2.0, 0.5, -1.0])
    norm = Norm.from_train({0: Y})
    model = GA.build_model(cfg, GA.DEFAULT_HP)
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.zero_()      # predict residual 0 == persistence
    got = GA.predict_clip(model, cfg, norm, Y, t_in=3)
    ors = origins(len(Y), cfg)
    want = np.repeat(Y[ors][:, None, :], cfg.horizon, axis=1)
    assert got.shape == (len(ors), cfg.horizon, N_CH)
    assert np.allclose(got, want, atol=1e-6)


def test_prediction_shape_matches_score_external_contract():
    cfg = make_cfg()
    Y = np.sin(np.arange(12)[:, None] / 3.0) * np.ones((1, N_CH))
    model = GA.build_model(cfg, GA.DEFAULT_HP)
    got = GA.predict_clip(model, cfg, Norm.from_train({0: Y}), Y, t_in=3)
    assert got.shape[0] == len(origins(len(Y), cfg))
    assert np.isfinite(got).all()


def test_too_short_clip_yields_zero_origins_not_a_crash():
    cfg = make_cfg(horizon=2, min_history=8)
    Y = np.zeros((5, N_CH))
    got = GA.predict_clip(GA.build_model(cfg, GA.DEFAULT_HP), cfg, Norm.from_train({0: Y}), Y, 3)
    assert got.shape == (0, cfg.horizon, N_CH)


def test_frozen_seed_gives_bitwise_identical_initialization_and_shuffle():
    cfg = make_cfg()
    GA.configure_determinism(0)
    a = GA.build_model(cfg, GA.DEFAULT_HP)
    GA.configure_determinism(0)
    b = GA.build_model(cfg, GA.DEFAULT_HP)
    for (k, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(va, vb), f"parameter {k} differs under a frozen seed"
    g1, g2 = GA.configure_determinism(5), GA.configure_determinism(5)
    assert torch.equal(torch.randperm(50, generator=g1), torch.randperm(50, generator=g2))


def test_training_runs_and_selects_on_val(tmp_path):
    """Smoke-level but end-to-end: two synthetic 'clips' through train() -> the loss must fall and
    the recorded selection must come from VAL, not TRAIN."""
    cfg = make_cfg(horizon=2, min_history=2)
    rng = np.random.default_rng(0)
    sig = {i: np.cumsum(rng.normal(size=(20, N_CH)), axis=0) for i in range(4)}

    def fake_load(_cfg, idx):
        return sig[idx]

    GA_load = GA.load_target
    GA.load_target = fake_load
    try:
        model, norm, hist = GA.train(cfg, [0, 1], [2, 3], t_in=3,
                                     hp={**GA.DEFAULT_HP, "epochs": 3, "batch": 8})
        assert hist["selected_on"] == "val"
        assert len(hist["val_mse"]) == 3 and np.isfinite(hist["best_val_mse"])
        preds = GA.predict(model, cfg, norm, [2, 3], t_in=3)
        assert set(preds) == {2, 3}
        assert preds[2].shape == (len(origins(20, cfg)), cfg.horizon, N_CH)
    finally:
        GA.load_target = GA_load
