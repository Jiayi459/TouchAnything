"""Clip-level bootstrap confidence intervals for the harness (OQ-D; Q4/Q5, 2026-08-12).

NEW MODULE (not a fork).

WHY CLIP-LEVEL AND NOT WINDOW-LEVEL (OQ-D, 2026-08-11): with stride 1 and a 1 s horizon at
30 Hz there are ~191k rolling windows over ~2,958 clips, and adjacent windows share almost all
of their history and target. They are nowhere near independent. Resampling windows would treat
strongly autocorrelated observations as independent draws and would badly understate the
variance. The approximately independent unit is the CLIP, so every interval here is built by
resampling clips (rows of a `ClipStats` table), never windows.

TWO DIFFERENT DESIGNS, because G1 and G2 compare different things (Q4, confirmed 2026-08-12):

  G1 — MODEL comparison (e.g. AR vs GRU-aggregate). Both models are scored on the SAME test
       clips, so the comparison is naturally PAIRED: each iteration draws one clip multiset and
       scores BOTH models on it, then takes the difference. The pairing removes the
       clip-difficulty variance that is common to both models, which is exactly the nuisance
       term that would otherwise dominate a small model gap. `bootstrap_paired` makes the
       sharing STRUCTURAL: the statistic function receives one row-index array and is the only
       thing that knows there are two models, so a paired design cannot be silently broken by
       drawing twice.

  G2 — CLASS comparison (dR2 = R2_smooth - R2_abrupt). smooth and abrupt are DISJOINT clip
       sets; no clip is both, so there is nothing to pair and no pairing is imposed
       artificially. `bootstrap_two_sample` resamples each class independently, within strata.

STRATIFICATION (G2, per the 2026-08-12 instruction "独立 stratified clip bootstrap"): each
class's resample is drawn WITHIN strata and preserves each stratum's clip count exactly. The
default intended stratum is ACTION, because the smooth class is dominated by a couple of
actions (`holding`/`sliding` are the bulk of it after the full-vocabulary audit) and an
unstratified draw could return a resample that is nearly all one action -- turning a
class-level interval into a single-action interval. The stratum labels are supplied by the
caller, so scene- or participant-level stratification needs no code change (SESSION_LOG
2026-08-12 records this as the one open question on Q4).

SEED / REPRODUCIBILITY (Q5): `numpy.random.default_rng` (PCG64), never the legacy global
`np.random.*`. `rng_provenance()` returns the numpy version and bit-generator name to be
stamped next to any reported interval -- a frozen seed only reproduces a number if the
generator behind it is also recorded.

B: 5000 for anything reported, 500 for development iteration (Q5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

B_FORMAL = 5000
B_DEV = 500
ALPHA = 0.05          # 95% interval from the 2.5 / 97.5 percentiles


def rng_provenance(seed: int, b: int) -> dict[str, object]:
    """Everything needed to reproduce an interval, for stamping into the results table."""
    return {"numpy_version": np.__version__,
            "bit_generator": type(np.random.default_rng(seed).bit_generator).__name__,
            "seed": int(seed), "B": int(b)}


def resample_rows(rng: np.random.Generator, n: int, strata=None) -> np.ndarray:
    """One with-replacement draw of `n` row indices.

    strata=None -> plain draw over 0..n-1. Otherwise `strata` is a length-n array of labels and
    each stratum is resampled to its OWN size, so the stratum composition of every resample
    equals that of the observed sample. Strata are visited in sorted label order so a given
    seed reproduces a given draw.
    """
    if strata is None:
        return rng.integers(0, n, size=n)
    strata = np.asarray(strata)
    if strata.shape[0] != n:
        raise ValueError(f"strata has length {strata.shape[0]}, expected {n}")
    parts = []
    for label in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == label)
        parts.append(idx[rng.integers(0, idx.size, size=idx.size)])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class BootstrapResult:
    """`point` is the statistic on the OBSERVED sample (not the mean of the resamples).

    Percentile interval. `bias` = mean(samples) - point is reported so a strongly skewed or
    biased resampling distribution is visible instead of hidden inside a symmetric-looking
    interval; if it is large relative to the interval width, say so rather than quoting the
    percentile CI as if it were exact.
    """
    point: np.ndarray
    samples: np.ndarray            # (B, D)
    lo: np.ndarray
    hi: np.ndarray
    alpha: float
    provenance: dict = field(default_factory=dict)

    @property
    def bias(self) -> np.ndarray:
        return np.nanmean(self.samples, axis=0) - self.point

    @property
    def excludes_zero(self) -> np.ndarray:
        """(D,) bool: the interval lies entirely above or entirely below 0."""
        return (self.lo > 0) | (self.hi < 0)


def percentile_ci(samples: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) from the alpha/2 and 1-alpha/2 percentiles, NaN-tolerant (a degenerate
    resample -- e.g. one where a channel had no valid clip -- must not poison the interval)."""
    s = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    lo = np.nanpercentile(s, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(s, 100 * (1 - alpha / 2), axis=0)
    return np.atleast_1d(lo), np.atleast_1d(hi)


def bootstrap_paired(stat_fn, n_clips: int, b: int = B_FORMAL, seed: int = 0,
                     strata=None, alpha: float = ALPHA) -> BootstrapResult:
    """PAIRED clip bootstrap (G1). `stat_fn(rows) -> (D,)` gets ONE row-index array per
    iteration and must score every model it compares on exactly those rows.

    The observed-sample point estimate is stat_fn(arange(n_clips))."""
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(stat_fn(np.arange(n_clips)), dtype=np.float64))
    out = np.empty((b, point.size), dtype=np.float64)
    for i in range(b):
        out[i] = np.atleast_1d(np.asarray(stat_fn(resample_rows(rng, n_clips, strata)),
                                          dtype=np.float64))
    lo, hi = percentile_ci(out, alpha)
    return BootstrapResult(point=point, samples=out, lo=lo, hi=hi, alpha=alpha,
                           provenance=rng_provenance(seed, b) | {"design": "paired"})


def bootstrap_two_sample(stat_fn, n_a: int, n_b: int, b: int = B_FORMAL, seed: int = 0,
                         strata_a=None, strata_b=None, alpha: float = ALPHA) -> BootstrapResult:
    """INDEPENDENT stratified two-sample clip bootstrap (G2 dR2).
    `stat_fn(rows_a, rows_b) -> (D,)`; the two groups are resampled independently, each within
    its own strata. No pairing is constructed -- the two clip sets are disjoint."""
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(stat_fn(np.arange(n_a), np.arange(n_b)), dtype=np.float64))
    out = np.empty((b, point.size), dtype=np.float64)
    for i in range(b):
        ra = resample_rows(rng, n_a, strata_a)
        rb = resample_rows(rng, n_b, strata_b)
        out[i] = np.atleast_1d(np.asarray(stat_fn(ra, rb), dtype=np.float64))
    lo, hi = percentile_ci(out, alpha)
    return BootstrapResult(point=point, samples=out, lo=lo, hi=hi, alpha=alpha,
                           provenance=rng_provenance(seed, b) | {"design": "two_sample"})
