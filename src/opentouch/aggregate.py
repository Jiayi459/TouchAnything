"""Per-clip sufficient statistics -> clip-balanced R2 / dR2 / skill.

NEW MODULE (not a fork). It implements the aggregation decisions taken on 2026-08-12 (Q1/Q2)
and is the single place where "how a class-level number is formed from clips" is defined.
metrics.py stays a byte-identical fork of the ActionSense scoring definitions; nothing here
changes MSE/MAE.

--------------------------------------------------------------------------------------------
Q2 — CLIP-BALANCED AGGREGATION (decided 2026-08-12)
--------------------------------------------------------------------------------------------
NOT the mean of per-clip R2 (a clip whose target happens to sit on the baseline has a
near-zero denominator, and its exploding ratio would dominate the average). NOT the plain sum
of raw SSE either (a 46 s clip would carry ~87x the weight of a 0.53 s clip, so the number
would describe the long clips). Instead:

    within each clip, divide SSE by that clip's VALID POINT COUNT  (=> per-clip MSE, so every
    clip carries total weight 1), then aggregate as a RATIO OF MEANS:

        R2_class = 1 - mean_over_clips(SSE_model_k / n_k) / mean_over_clips(SSE_base_k / n_k)

`n_k` is counted PER CHANNEL, because CoP channels are masked at low force (masking.py) while
force never is -- a clip can contribute 900 valid force points and 40 valid CoP points. A clip
with zero valid points in a channel is dropped from THAT channel only, and the same clip set
is used in numerator and denominator so the ratio is never mixing populations.

Note the counted unit is a (origin, horizon-step) pair, not a unique frame: at stride 1 and
H=30 each frame is a target in up to 30 windows. That is the harness's existing convention
(metrics.py aggregates over N*H) and is kept so R2 and MSE describe the same point set.

--------------------------------------------------------------------------------------------
Q1 — WHICH MEAN IS THE R2 DENOMINATOR (decided 2026-08-12, with a correction)
--------------------------------------------------------------------------------------------
Primary = `class_mean`: the clip-balanced per-channel mean of the TEST subset being scored.
This is standard R2 (variance explained relative to the sample's own mean).

The 2026-08-11 plan asserted that a TEST-derived mean "leaks TEST information into the
baseline, so the number comes out inflated". THAT WAS WRONG IN DIRECTION, and the record is
corrected here rather than quietly dropped: the sample mean MINIMIZES squared error on the
sample it is computed from, so SSE_base(class_mean) <= SSE_base(train_mean), hence

        R2(class_mean)  <=  R2(train_mean) == R2_OS

i.e. the TEST-mean denominator is the STRICTER of the two, not the flattering one. The user's
reasoning is also correct on its own terms: the mean is applied AFTER the model has produced
its predictions and is never visible to the model, so it cannot leak into a forecast.

Secondary = `train_mean` (R2_OS / train-mean skill, Campbell-Thompson style): a genuine
out-of-sample predictive-skill statement -- could a forecaster who knew only the TRAIN mean
have done better? Reported as robustness, never as "R2".

Also available = `clip_mean`: each clip against ITS OWN mean, i.e. within-clip variance
explained. This exists because `class_mean` has a real interpretive hazard for this corpus:
clips are short (2.8 s median) with heterogeneous DC force levels, so a single class-wide
constant makes BETWEEN-clip level variance dominate SST, and any model that merely tracks the
current level (persistence does, for free) scores a high R2 without explaining any dynamics.
`clip_mean` isolates the within-clip dynamics that the trait hypothesis is actually about.
Which of the two should be primary is logged as an OPEN QUESTION (SESSION_LOG 2026-08-12);
both are computed from the same tables at no extra cost, so the choice is not load-bearing
for the implementation.

--------------------------------------------------------------------------------------------
WHY SUFFICIENT STATISTICS AND NOT JUST SSE
--------------------------------------------------------------------------------------------
Per clip and channel we store (n_valid, sum_y, sum_y2) plus one SSE column per predictor.
For ANY constant baseline mu, SSE = sum_y2 - 2*mu*sum_y + n*mu^2 in closed form. Consequences:
  * a bootstrap resample can recompute ITS OWN clip-balanced mean instead of inheriting the
    full-sample mean -- the mean is part of the statistic being resampled, so treating it as
    fixed would understate the variance;
  * the Layer-3 sensitivity analysis (drop the contentious actions and recompute) is table
    algebra on a few hundred rows -- no model refitting, no re-rolling of origins;
  * per-clip MSE, R2 against three different baselines, dR2 and skill all come out of one
    pass over the prediction tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12

CLASS_MEAN = "class_mean"
TRAIN_MEAN = "train_mean"
CLIP_MEAN = "clip_mean"
BASELINES = (CLASS_MEAN, TRAIN_MEAN, CLIP_MEAN)


@dataclass(frozen=True)
class ClipStats:
    """Per-clip x per-channel sufficient statistics for one scored population.

    clip_ids : (K,)   ascending clip idxs
    n_valid  : (K,C)  count of unmasked (origin, horizon-step) target points
    sum_y    : (K,C)  sum of valid targets            } enough to rebuild the SSE of ANY
    sum_y2   : (K,C)  sum of squared valid targets    } constant predictor, in closed form
    sse      : predictor name -> (K,C) sum of squared errors over the valid points
    """
    clip_ids: np.ndarray
    n_valid: np.ndarray
    sum_y: np.ndarray
    sum_y2: np.ndarray
    sse: dict[str, np.ndarray]
    channels: tuple[str, ...]

    @property
    def n_clips(self) -> int:
        return int(self.clip_ids.size)

    @property
    def n_channels(self) -> int:
        return int(self.n_valid.shape[1])

    def rows_of(self, clip_ids) -> np.ndarray:
        """Row indices for the given clip idxs (for subsetting, e.g. dropping contentious)."""
        want = np.asarray(sorted(set(int(c) for c in clip_ids)), dtype=np.int64)
        pos = np.searchsorted(self.clip_ids, want)
        ok = (pos < self.clip_ids.size) & (self.clip_ids[np.minimum(pos, self.clip_ids.size - 1)] == want)
        return pos[ok]

    def drop(self, clip_ids) -> np.ndarray:
        """Row indices EXCLUDING the given clip idxs (the Layer-3 sensitivity subset)."""
        keep = np.ones(self.clip_ids.size, dtype=bool)
        keep[self.rows_of(clip_ids)] = False
        return np.flatnonzero(keep)


def clip_stats(ytrue: np.ndarray, mask: np.ndarray, clip_ids: np.ndarray,
               preds: dict[str, np.ndarray], channels=()) -> ClipStats:
    """Collapse (N,H,C) prediction tensors into per-clip sufficient statistics.

    ytrue/mask/each preds[name] are (N,H,C) aligned on the origin axis; clip_ids is (N,)
    from baselines.predict_series_by_clip. Masked-out points are excluded from every
    accumulator, so all predictors and all baselines are scored on an IDENTICAL point set
    (the mask depends only on target force, never on the model -- masking.py).
    """
    ytrue = np.asarray(ytrue, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    clip_ids = np.asarray(clip_ids, dtype=np.int64)
    if ytrue.shape != m.shape:
        raise ValueError(f"mask shape {m.shape} != ytrue shape {ytrue.shape}")
    if clip_ids.shape[0] != ytrue.shape[0]:
        raise ValueError("clip_ids must align with the origin axis of ytrue")

    C = ytrue.shape[-1]
    uniq, row = np.unique(clip_ids, return_inverse=True)      # uniq ascending
    K = uniq.size

    def group(per_window: np.ndarray) -> np.ndarray:
        """(N,C) -> (K,C) summed within clip."""
        out = np.zeros((K, C), dtype=np.float64)
        for c in range(C):
            out[:, c] = np.bincount(row, weights=per_window[:, c], minlength=K)
        return out

    n_valid = group(m.sum(1))                                  # sum over horizon steps
    sum_y = group((ytrue * m).sum(1))
    sum_y2 = group((ytrue ** 2 * m).sum(1))
    sse = {}
    for name, yhat in preds.items():
        yhat = np.asarray(yhat, dtype=np.float64)
        if yhat.shape != ytrue.shape:
            raise ValueError(f"preds[{name!r}] shape {yhat.shape} != ytrue shape {ytrue.shape}")
        sse[name] = group(((yhat - ytrue) ** 2 * m).sum(1))
    return ClipStats(clip_ids=uniq, n_valid=n_valid.astype(np.int64), sum_y=sum_y,
                     sum_y2=sum_y2, sse=sse,
                     channels=tuple(channels) if channels else tuple(f"c{i}" for i in range(C)))


# --------------------------------------------------------------------------- baselines --
def _rows(st: ClipStats, rows) -> np.ndarray:
    return np.arange(st.n_clips) if rows is None else np.asarray(rows, dtype=np.int64)


def clip_balanced_mean(st: ClipStats, rows=None) -> np.ndarray:
    """(C,): the unweighted mean over clips of each clip's own mean target.

    This is EXACTLY the constant that minimizes the clip-balanced objective
    mean_k(SSE_k(mu)/n_k) -- d/dmu gives mu = (1/K) * sum_k mean_k(y), the average of per-clip
    means, NOT the point-weighted grand mean. Using the point-weighted mean instead would make
    the "mean baseline" a non-minimizer under the weighting we actually aggregate with, so a
    model could beat it without explaining anything. Clips with no valid points in a channel
    are excluded from that channel.
    """
    r = _rows(st, rows)
    n = st.n_valid[r].astype(np.float64)
    ok = n > 0
    per_clip = np.divide(st.sum_y[r], n, out=np.zeros_like(n), where=ok)
    cnt = ok.sum(0)
    tot = (per_clip * ok).sum(0)
    return np.divide(tot, cnt, out=np.full(st.n_channels, np.nan), where=cnt > 0)


def constant_sse(st: ClipStats, mu: np.ndarray, rows=None) -> np.ndarray:
    """(K',C) SSE of the constant predictor `mu` (per channel), in closed form from the
    sufficient statistics: sum(y-mu)^2 = sum_y2 - 2*mu*sum_y + n*mu^2."""
    r = _rows(st, rows)
    mu = np.asarray(mu, dtype=np.float64).reshape(1, -1)
    return st.sum_y2[r] - 2.0 * mu * st.sum_y[r] + st.n_valid[r] * mu ** 2


def within_clip_sse(st: ClipStats, rows=None) -> np.ndarray:
    """(K',C) SSE of each clip against ITS OWN mean = within-clip SST. Clipped at 0 because
    sum_y2 - sum_y^2/n can go slightly negative through floating-point cancellation."""
    r = _rows(st, rows)
    n = st.n_valid[r].astype(np.float64)
    ok = n > 0
    val = st.sum_y2[r] - np.divide(st.sum_y[r] ** 2, n, out=np.zeros_like(n), where=ok)
    return np.where(ok, np.maximum(val, 0.0), 0.0)


def baseline_sse(st: ClipStats, baseline: str, rows=None, train_mean=None) -> np.ndarray:
    """(K',C) SSE of the requested R2 denominator. `train_mean` (C,) is required for
    TRAIN_MEAN and must be the clip-balanced mean of the TRAIN split (dataset-level, frozen)."""
    if baseline == CLASS_MEAN:
        return constant_sse(st, clip_balanced_mean(st, rows), rows)
    if baseline == TRAIN_MEAN:
        if train_mean is None:
            raise ValueError("baseline='train_mean' needs the TRAIN clip-balanced mean")
        return constant_sse(st, train_mean, rows)
    if baseline == CLIP_MEAN:
        return within_clip_sse(st, rows)
    raise ValueError(f"unknown baseline {baseline!r}; expected one of {BASELINES}")


# ------------------------------------------------------------------------ aggregation --
def clip_equal_ratio(num_sse: np.ndarray, den_sse: np.ndarray, n_valid: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Ratio of clip-balanced means: mean_k(num_k/n_k) / mean_k(den_k/n_k), per channel.

    -> (ratio (C,), n_clips_used (C,)). Clips with n_valid == 0 in a channel are dropped from
    that channel in BOTH numerator and denominator (identical clip set, so the ratio compares
    like with like). A channel with no usable clip, or a vanishing denominator, yields NaN --
    propagated deliberately rather than silently turned into 0 or 1.
    """
    n = np.asarray(n_valid, dtype=np.float64)
    ok = n > 0
    num = np.divide(num_sse, n, out=np.zeros_like(n), where=ok)
    den = np.divide(den_sse, n, out=np.zeros_like(n), where=ok)
    cnt = ok.sum(0)
    C = n.shape[1]
    num_m = np.divide((num * ok).sum(0), cnt, out=np.full(C, np.nan), where=cnt > 0)
    den_m = np.divide((den * ok).sum(0), cnt, out=np.full(C, np.nan), where=cnt > 0)
    ratio = np.divide(num_m, den_m, out=np.full(C, np.nan),
                      where=np.isfinite(den_m) & (np.abs(den_m) > EPS))
    return ratio, cnt.astype(np.int64)


@dataclass(frozen=True)
class R2Result:
    """Per-channel R2 is the RESULT; `headline` is a reading convenience only.

    A single averaged number can hide the story the trait hypothesis is about (a CoP-only or
    force-only effect averages away), so conclusions are drawn from `per_channel` and the
    headline is never reported alone -- see Q1, SESSION_LOG 2026-08-12.
    """
    per_channel: np.ndarray      # (C,)
    n_clips: np.ndarray          # (C,) clips contributing per channel
    channels: tuple[str, ...]
    model: str
    baseline: str

    @property
    def headline(self) -> float:
        return float(np.nanmean(self.per_channel))

    def as_dict(self) -> dict[str, float]:
        return {ch: float(v) for ch, v in zip(self.channels, self.per_channel)}


def r2(st: ClipStats, model: str, baseline: str = CLASS_MEAN, rows=None,
       train_mean=None) -> R2Result:
    """Clip-balanced R2 of `model` against the chosen mean baseline. `rows` selects a clip
    subset (a bootstrap resample, or the contentious-dropped set)."""
    if model not in st.sse:
        raise KeyError(f"no SSE column for model {model!r}; have {sorted(st.sse)}")
    r = _rows(st, rows)
    num = st.sse[model][r]
    den = baseline_sse(st, baseline, rows=r, train_mean=train_mean)
    ratio, cnt = clip_equal_ratio(num, den, st.n_valid[r])
    return R2Result(per_channel=1.0 - ratio, n_clips=cnt, channels=st.channels,
                    model=model, baseline=baseline)


def skill(st: ClipStats, model: str, ref: str, rows=None) -> np.ndarray:
    """(C,) clip-balanced 1 - MSE_model/MSE_ref against another PREDICTOR (not a mean).
    Diagnostic only for G2 (OQ-A: skill-vs-persistence does not participate in inference)."""
    r = _rows(st, rows)
    ratio, _ = clip_equal_ratio(st.sse[model][r], st.sse[ref][r], st.n_valid[r])
    return 1.0 - ratio


def delta_r2(a: R2Result, b: R2Result) -> np.ndarray:
    """(C,) R2(a) - R2(b), e.g. dR2 = R2_smooth - R2_abrupt.

    CAVEAT to carry into the write-up: with `class_mean`, the two classes have DIFFERENT
    denominators (each class's own within-class variance), which is what "class-specific R2"
    means but does make dR2 sensitive to a difference in class variance as well as to a
    difference in forecastability. That is why OQ-A already puts raw MSE/MAE alongside as
    secondary evidence, and why the `clip_mean` variant is worth reporting.
    """
    if a.channels != b.channels:
        raise ValueError("dR2 requires matching channel orders")
    if a.baseline != b.baseline:
        raise ValueError(f"dR2 across different baselines ({a.baseline} vs {b.baseline})")
    return a.per_channel - b.per_channel
