"""Frozen evaluation entry point.

Protocol (constraint 2, no leakage):
  1. Load the frozen split (splits.json). 2. Fit Norm + force thresholds on TRAIN only.
  3. Each baseline: fit(TRAIN) -> select hyperparams on VAL -> forecast TEST (touched once).
  4. Score with the frozen metrics + CoP mask. 5. Write a results table stamped with the
  config hash. Determinism: the whole thing is recomputed twice and asserted identical.

    python -m src.tactile_forecast.eval_harness.evaluate
    python -m src.tactile_forecast.eval_harness.evaluate --rebuild-splits
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from . import baselines as BL
from . import masking, metrics
from .config import load_config, Config
from .dataset import Norm, force_thresholds, load_group
from .splits import load_splits


def run_once(cfg: Config, splits: dict) -> dict:
    """Fit/select/score every baseline on the frozen split. Returns per-baseline arrays."""
    train = load_group(cfg, splits["train"])
    val = load_group(cfg, splits["val"])
    test = load_group(cfg, splits["test"])
    norm = Norm.from_train(train)
    thr = force_thresholds(cfg, train)

    # reference (persistence) first — needed for skill
    results = {}
    ref_mse = None
    for cls in [BL.Persistence, BL.SeasonalNaive, BL.AR]:
        bl = cls(cfg, norm)
        bl.fit(train)
        bl.select(val, cfg.horizon)
        ytrue, yhat = BL.predict_series(bl, test, cfg)
        mask = masking.valid_mask(cfg, ytrue.reshape(-1, 6), thr).reshape(ytrue.shape)
        ch_mse = metrics.masked_channel_mse(ytrue, yhat, mask)
        hz_mse = metrics.masked_horizon_mse(ytrue, yhat, mask)
        ch_mae = metrics.masked_channel_mae(ytrue, yhat, mask)
        results[bl.name] = {
            "ch_mse": ch_mse, "hz_mse": hz_mse, "ch_mae": ch_mae,
            "n_valid": mask.reshape(-1, 6).sum(0),
            "hyper": getattr(bl, "order", getattr(bl, "period", None)),
            "nrmse": metrics.normalized_rmse(ch_mse, norm.std),
        }
        if bl.name == "persistence":
            ref_mse = {"ch": ch_mse, "hz": hz_mse}
    for name, r in results.items():
        r["ch_skill"] = metrics.skill(r["ch_mse"], ref_mse["ch"])
        r["hz_skill"] = metrics.skill(r["hz_mse"], ref_mse["hz"])
    return results


def _flatten(results: dict) -> np.ndarray:
    """Deterministic numeric fingerprint for the identical-runs assertion."""
    keys = sorted(results)
    return np.concatenate([results[k]["hz_mse"].ravel() for k in keys]
                          + [results[k]["ch_mse"].ravel() for k in keys])


def write_csv(cfg: Config, splits: dict, results: dict, out: str) -> None:
    chans = cfg.channels
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_hash", "baseline", "hyper", "channel", "horizon_step_s",
                    "mse", "rmse", "mae", "skill_vs_persistence", "n_valid"])
        for name in ["persistence", "seasonal", "ar"]:
            r = results[name]
            for ci, ch in enumerate(chans):
                for h in range(cfg.horizon):
                    mse = r["hz_mse"][h, ci]
                    w.writerow([cfg.config_hash, name, r["hyper"], ch,
                                round((h + 1) / cfg.fps, 3), f"{mse:.6g}",
                                f"{np.sqrt(mse):.6g}",
                                f"{r['ch_mae'][ci]:.6g}",
                                f"{r['hz_skill'][h, ci]:.4f}", int(r["n_valid"][ci])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    splits = load_splits(cfg, rebuild=args.rebuild_splits)
    print(f"config_hash={cfg.config_hash}  fps={cfg.fps:.0f}  horizon={cfg.horizon} steps  "
          f"split n={splits['n']} (train {len(splits['train'])}, val {len(splits['val'])}, "
          f"test {len(splits['test'])})")

    results = run_once(cfg, splits)
    # determinism: identical second run
    fp1 = _flatten(results)
    fp2 = _flatten(run_once(cfg, splits))
    assert np.array_equal(fp1, fp2), "NON-DETERMINISTIC: two runs differ"
    print("determinism check: PASS (two runs identical)")

    out = args.out or cfg.abspath("out_csv")
    write_csv(cfg, splits, results, out)
    # console summary
    print(f"\n{'baseline':12}{'hyper':>7}{'nRMSE':>9}   mean skill vs persistence (per channel)")
    for name in ["persistence", "seasonal", "ar"]:
        r = results[name]
        sk = "  ".join(f"{c}:{s:+.2f}" for c, s in zip(cfg.channels, r["ch_skill"]))
        print(f"{name:12}{str(r['hyper']):>7}{r['nrmse']:>9.3f}   {sk}")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
