"""Frozen evaluation entry point — fork of src/actionsense/eval_harness/evaluate.py.

FORKED, NOT SHARED (see masking.py's docstring). Changes from the original:
  1. Every hardcoded 6 -> len(cfg.channels) (this module scores a 3-channel target).
  2. Imports point at src.opentouch's own masking/metrics/baselines/dataset (the forks
     and the new OpenTouch dataset module), not src.actionsense.eval_harness's.
  3. `main()` does NOT import or call a splits loader: src/opentouch/splits.py does not
     exist yet (blocked — see configs/opentouch/eval_harness.yaml's "split" block and
     SESSION_LOG 2026-08-10 for why). Calling `main()` raises NotImplementedError with
     that reason. Every other function here works today given a caller-supplied `splits`
     dict (e.g. from a unit test), so this module is fully testable without it.

Protocol (unchanged): load the split -> fit Norm + force thresholds on TRAIN -> each
baseline fit(TRAIN) then select hyperparameters on VAL -> forecast TEST (touched once) ->
score with the frozen metrics + CoP mask. Skill scores are computed on IDENTICAL masked
frame sets (the mask depends only on target-frame force, not the model).

Standard prediction format for scoring EXTERNAL models (e.g. the GRU) against this harness:
a dict {clip_idx: yhat} where yhat has shape (n_origins, H, C) and n_origins/order match
baselines.origins(len(Y), cfg) for that clip (target-time indexed, h = 1..H).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import baselines as BL
from . import masking, metrics
from .dataset import Norm, force_thresholds, group_keys, load_group
from src.actionsense.eval_harness.config import load_config, Config

MODELS = ["persistence", "seasonal", "ar"]
CLASSES = {"persistence": BL.Persistence, "seasonal": BL.SeasonalNaive, "ar": BL.AR}


def _hand(channel: str) -> str:
    # All OpenTouch channels end in "_R" (right hand only) -- kept for structural parity
    # with the ActionSense original rather than removed, so the row schema matches.
    return "L" if channel.endswith("_L") else "R"


def _result(ytrue, yhat, mask) -> dict:
    C = mask.shape[-1]
    return {
        "hz_mse": metrics.masked_horizon_mse(ytrue, yhat, mask),
        "hz_mae": metrics.masked_horizon_mae(ytrue, yhat, mask),
        "ch_mse": metrics.masked_channel_mse(ytrue, yhat, mask),
        "ch_mae": metrics.masked_channel_mae(ytrue, yhat, mask),
        "n": mask.reshape(-1, C).sum(0).astype(int),
    }


def fit_and_forecast(cfg: Config, splits: dict):
    """Fit/select every baseline; return (results, mask, extras). results[name] has masked
    metric arrays; extras carries the seasonal periods."""
    train, val, test = (load_group(cfg, splits[k]) for k in ("train", "val", "test"))
    gtr = group_keys(cfg, splits["train"]); gva = group_keys(cfg, splits["val"])
    gte = group_keys(cfg, splits["test"])
    norm = Norm.from_train(train)
    thr = force_thresholds(cfg, train)
    C = len(cfg.channels)

    results, mask, extras = {}, None, {}
    for name in MODELS:
        bl = CLASSES[name](cfg, norm)
        bl.fit(train, gtr)
        bl.select(val, gva, cfg.horizon)
        ytrue, yhat = BL.predict_series(bl, test, gte, cfg)
        if mask is None:
            mask = masking.valid_mask(cfg, ytrue.reshape(-1, C), thr).reshape(ytrue.shape)
        results[name] = _result(ytrue, yhat, mask)
        if name == "seasonal":
            extras["seasonal_periods"] = dict(bl.periods)
            extras["ar_orders"] = None
        if name == "ar":
            extras["ar_orders"] = dict(bl.order)
    return results, norm, extras


def build_rows(cfg: Config, results: dict) -> list[dict]:
    """Tidy long rows: [model, channel, hand, horizon_step, metric, value, n_frames, config_hash]."""
    rows = []
    H, chans = cfg.horizon, cfg.channels

    def emit(model, ci, step, metric, value, n):
        rows.append({"model": model, "channel": chans[ci], "hand": _hand(chans[ci]),
                     "horizon_step": str(step), "metric": metric, "value": float(value),
                     "n_frames": int(n), "config_hash": cfg.config_hash})

    for m in MODELS:
        R = results[m]
        for ci in range(len(chans)):
            n = R["n"][ci]
            for h in range(H):
                emit(m, ci, h + 1, "MSE", R["hz_mse"][h, ci], n)
                emit(m, ci, h + 1, "MAE", R["hz_mae"][h, ci], n)
                for b in MODELS:
                    ss = metrics.skill(R["hz_mse"][h, ci], results[b]["hz_mse"][h, ci])
                    emit(m, ci, h + 1, f"SS_vs_{b}", ss, n)
            emit(m, ci, "all", "MSE", R["ch_mse"][ci], n)
            emit(m, ci, "all", "MAE", R["ch_mae"][ci], n)
            for b in MODELS:
                ss = metrics.skill(R["ch_mse"][ci], results[b]["ch_mse"][ci])
                emit(m, ci, "all", f"SS_vs_{b}", ss, n)
    return rows


def score_external(cfg: Config, splits: dict, name: str, preds: dict[int, np.ndarray],
                   ref_results: dict, norm: Norm) -> dict:
    """Score an external model's predictions against the frozen baselines. `preds[idx]` is
    (n_origins, H, C) aligned to baselines.origins(len(Y), cfg)."""
    C = len(cfg.channels)
    test = load_group(cfg, splits["test"])
    thr = force_thresholds(cfg, load_group(cfg, splits["train"]))
    yts, yhs = [], []
    for i, Y in sorted(test.items()):
        ors = BL.origins(len(Y), cfg)
        if i not in preds or preds[i].shape[0] != len(ors):
            raise ValueError(f"preds[{i}] must have shape ({len(ors)}, {cfg.horizon}, {C})")
        for j, t in enumerate(ors):
            yts.append(Y[t + 1:t + 1 + cfg.horizon]); yhs.append(preds[i][j])
    ytrue, yhat = np.stack(yts), np.stack(yhs)
    mask = masking.valid_mask(cfg, ytrue.reshape(-1, C), thr).reshape(ytrue.shape)
    return _result(ytrue, yhat, mask)


def write_table(rows: list[dict], out_csv: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_parquet(os.path.splitext(out_csv)[0] + ".parquet", index=False)
    return df


def _fingerprint(results: dict) -> np.ndarray:
    return np.concatenate([results[m]["hz_mse"].ravel() for m in MODELS]
                          + [results[m]["ch_mse"].ravel() for m in MODELS])


def main():
    raise NotImplementedError(
        "src/opentouch/splits.py does not exist yet -- whether OpenTouch's '_p1'/'_p2' "
        "shard suffix denotes participant or recording session is unresolved (see "
        "configs/opentouch/eval_harness.yaml 'split' block and SESSION_LOG 2026-08-10). "
        "fit_and_forecast()/build_rows()/score_external() all work today given a "
        "caller-supplied `splits` dict; only the CLI entry point is blocked."
    )


if __name__ == "__main__":
    main()
