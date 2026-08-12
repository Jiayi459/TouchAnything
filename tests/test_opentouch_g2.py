"""Unit tests for the G2 stage-1 modules: trait.py, aggregate.py, bootstrap.py, gru_aggregate.py.

All synthetic — no OpenTouch shard is required (the corpus is still downloading). The point of
these tests is NOT "it runs": each one pins a STATISTICAL PROPERTY that the 2026-08-12 design
decisions claim, so that a later refactor cannot quietly change the meaning of a reported number.

Notably pinned here:
  * the clip-balanced mean is the exact MINIMIZER of the clip-balanced objective (so "R2 = 0 iff
    the model equals the mean baseline" is true under our weighting, not approximately true);
  * R2(train_mean) >= R2(class_mean) — the direction correction recorded in aggregate.py's
    docstring, i.e. the TEST-mean denominator is the stricter one, not the flattering one;
  * long and short clips carry EQUAL weight (Q2);
  * ratio-of-means stays finite where mean-of-ratios explodes (the reason for Q2's design);
  * paired resampling shares indices STRUCTURALLY and therefore gives a tighter interval on a
    correlated difference than independent resampling would;
  * stratified resampling preserves each stratum's clip count exactly;
  * a GRU-aggregate whose head outputs 0 reproduces persistence EXACTLY (the residual-target
    convention), and its predictions align 1:1 with baselines.origins.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.actionsense.eval_harness.config import Config
from src.opentouch import aggregate as AG
from src.opentouch import bootstrap as BS
from src.opentouch import trait as TR

N_CH = 3


def make_cfg(horizon=2, min_history=2, stride=1, fps=30.0):
    raw = {
        "target": {"channels": ["F_R", "CoPx_R", "CoPy_R"], "force_idx": [0], "cop_idx": [1, 2]},
        "rate": {"fps_raw": fps, "downsample": 1, "horizon_s": horizon / fps},
        "mask": {"percentile": 5},
        "baselines": {"fit_scope": "object_category", "ar_orders": [2],
                      "seasonal_period_min_s": 0.2, "seasonal_period_max_s": 1.0,
                      "seasonal_min_autocorr": 0.1},
        "eval": {"stride": stride, "min_history": min_history, "seed": 0},
    }
    return Config(raw=raw, path="test", config_hash="test")


# ==================================================================== trait.py (Layer 1) ==
def test_audit_table_is_internally_consistent():
    assert set(TR.TRAIT_CLASS.values()) == {TR.SMOOTH, TR.ABRUPT}
    assert TR.CONTENTIOUS <= set(TR.TRAIT_CLASS)          # no contentious action left unaudited
    for key in TR.TRAIT_CLASS:                            # keys must already be canonical
        assert TR.normalize_action(key) == key


def test_users_explicit_rulings_are_encoded():
    """The 2026-08-12 rulings, verbatim. If a refactor flips one of these the test fails loudly."""
    for a in ("cutting", "flipping"):
        assert TR.trait_class(a) == TR.ABRUPT
    for a in ("scraping", "pouring", "stirring", "spreading", "drawing", "writing", "carrying"):
        assert TR.trait_class(a) == TR.SMOOTH
    # full-vocabulary audit outcome (user-approved scope): these two moved INTO smooth
    for a in ("holding", "sliding"):
        assert TR.trait_class(a) == TR.SMOOTH
    # user-flagged mixed-sub-event actions: abrupt under the strict rubric AND contentious
    for a in ("eating", "drinking", "scooping", "serving"):
        assert TR.trait_class(a) == TR.ABRUPT and TR.is_contentious(a)


def test_unaudited_and_missing_actions_raise_instead_of_defaulting():
    with pytest.raises(TR.UnauditedAction):
        TR.trait_class("juggling")            # not in the audited 30
    with pytest.raises(TR.UnauditedAction):
        TR.trait_class("")                    # missing annotation is not a class
    with pytest.raises(TR.UnauditedAction):
        TR.trait_class(None)


def test_normalize_action_handles_manifest_noise():
    assert TR.normalize_action("  Picking   Up ") == "picking up"
    assert TR.trait_class("PICKING UP") == TR.ABRUPT


def test_partition_buckets_everything_and_counts_drops():
    actions = {1: "holding", 2: "picking up", 3: "", 4: "juggling", 5: "cutting"}
    p = TR.partition(actions)
    assert p[TR.SMOOTH] == [1] and p[TR.ABRUPT] == [2, 5]
    assert p["unlabeled"] == [3] and p["unaudited"] == [4]
    assert sum(len(v) for v in p.values()) == len(actions)      # nothing vanishes
    assert TR.unaudited(actions.values()) == {"juggling"}
    assert TR.contentious_ids(actions) == [5]


# ============================================================ trait.py (Layer 2 statistics) ==
def test_causal_smooth_is_strictly_causal():
    x = np.arange(10.0)
    a = TR.causal_smooth(x, 3)
    y = x.copy(); y[7] = 999.0                       # corrupt the FUTURE
    b = TR.causal_smooth(y, 3)
    assert np.allclose(a[:7], b[:7])                 # earlier outputs unchanged
    assert a.shape == x.shape
    assert np.allclose(TR.causal_smooth(x, 1), x)    # k=1 is the identity


def test_delta_f_p95_separates_impulsive_from_ramp():
    n = 200
    ramp = np.linspace(0.0, 10.0, n)                                  # continuous modulation
    impulsive = np.zeros(n); impulsive[::20] = 10.0                   # discrete events
    assert TR.delta_f_p95(impulsive) > 10 * TR.delta_f_p95(ramp)


def test_hf_energy_fraction_separates_fast_from_slow():
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    slow = np.sin(2 * np.pi * 0.5 * t)
    fast = np.sin(2 * np.pi * 10.0 * t)
    assert TR.hf_energy_fraction(fast, fps) > 0.9
    assert TR.hf_energy_fraction(slow, fps) < 0.1


def test_short_clips_are_excluded_not_zero_filled():
    short = np.zeros(TR.MIN_DIFFS)                   # MIN_DIFFS-1 diffs: one too few
    assert TR.delta_f_p95(short) is None
    assert TR.hf_energy_fraction(short, 30.0) is None


def test_per_action_stat_reports_spread_and_drops():
    values = {1: 1.0, 2: 3.0, 3: None, 4: 7.0}
    actions = {1: "holding", 2: "holding", 3: "holding", 4: "cutting"}
    out = TR.per_action_stat(values, actions)
    assert out["holding"]["median"] == 2.0
    assert out["holding"]["n_clips"] == 2 and out["holding"]["n_dropped"] == 1
    assert out["holding"]["trait"] == TR.SMOOTH and out["holding"]["contentious"] is False
    assert out["cutting"]["contentious"] is True
    assert out["holding"]["min"] == 1.0 and out["holding"]["max"] == 3.0


# ======================================================================== aggregate.py ==
def _stats(clips, mask_map=None, preds_map=None, channels=("F_R", "CoPx_R", "CoPy_R")):
    """Build a ClipStats from {clip_id: (n_origins,H,C) ytrue}. preds_map: name -> same shape."""
    ids, ys = [], []
    for cid, Y in sorted(clips.items()):
        ys.append(np.asarray(Y, dtype=np.float64))
        ids += [cid] * len(Y)
    ytrue = np.concatenate(ys, 0)
    clip_ids = np.asarray(ids, dtype=np.int64)
    mask = np.ones_like(ytrue, dtype=bool) if mask_map is None else np.concatenate(
        [np.asarray(mask_map[c], dtype=bool) for c in sorted(clips)], 0)
    preds = {} if preds_map is None else {
        k: np.concatenate([np.asarray(v[c], dtype=np.float64) for c in sorted(clips)], 0)
        for k, v in preds_map.items()}
    return AG.clip_stats(ytrue, mask, clip_ids, preds, channels=channels), ytrue, clip_ids


def test_sufficient_statistics_reproduce_brute_force_sse():
    rng = np.random.default_rng(0)
    clips = {5: rng.normal(size=(4, 2, N_CH)), 9: rng.normal(size=(7, 2, N_CH))}
    st, ytrue, clip_ids = _stats(clips)
    mu = np.array([0.3, -0.2, 1.1])
    got = AG.constant_sse(st, mu)
    for r, cid in enumerate(st.clip_ids):
        sel = clip_ids == cid
        want = ((ytrue[sel] - mu) ** 2).reshape(-1, N_CH).sum(0)
        assert np.allclose(got[r], want)
    assert np.array_equal(st.clip_ids, np.array([5, 9]))
    assert np.array_equal(st.n_valid[:, 0], np.array([8, 14]))     # n_origins * H


def test_clip_balanced_mean_is_the_exact_minimizer():
    """Because the aggregate weights every clip equally, the minimizing constant is the average
    of per-clip means, NOT the point-weighted grand mean. Clips of different lengths make the
    two differ, so this test would fail if the wrong mean were used."""
    clips = {1: np.full((2, 2, N_CH), 0.0), 2: np.full((8, 2, N_CH), 10.0)}   # 8 vs 32 points
    st, ytrue, _ = _stats(clips)
    mu = AG.clip_balanced_mean(st)
    assert np.allclose(mu, 5.0)                                    # NOT the grand mean (8.0)
    assert not np.allclose(mu, ytrue.reshape(-1, N_CH).mean(0))

    def objective(m):
        sse = AG.constant_sse(st, np.full(N_CH, m))
        return float(np.mean(sse[:, 0] / st.n_valid[:, 0]))

    base = objective(5.0)
    for delta in (-1.0, -0.1, 0.1, 1.0):
        assert objective(5.0 + delta) > base


def test_long_and_short_clips_get_equal_weight():
    """A 4x longer clip must not carry 4x the weight (Q2). Model is perfect on the long clip and
    wrong on the short one; the class R2 must reflect a 50/50 mix of the two clips' MSEs."""
    short = np.zeros((2, 1, N_CH)); long = np.ones((8, 1, N_CH))
    preds = {"model": {1: short + 1.0, 2: long.copy()}}            # err 1.0 on clip 1, 0 on clip 2
    st, _, _ = _stats({1: short, 2: long}, preds_map=preds)
    mse = st.sse["model"] / st.n_valid
    assert np.allclose(mse[:, 0], [1.0, 0.0])
    num, cnt = AG.clip_equal_ratio(st.sse["model"], AG.constant_sse(st, AG.clip_balanced_mean(st)),
                                  st.n_valid)
    # numerator = mean(1.0, 0.0) = 0.5 ; denominator = mean over clips of SSE(mu=0.5)/n = 0.25
    assert np.allclose(num, 2.0)
    assert np.array_equal(cnt, np.full(N_CH, 2))


def test_r2_is_zero_for_the_mean_baseline_and_one_for_a_perfect_model():
    rng = np.random.default_rng(1)
    clips = {1: rng.normal(size=(5, 2, N_CH)), 2: rng.normal(size=(3, 2, N_CH))}
    st0, _, _ = _stats(clips)
    mu = AG.clip_balanced_mean(st0)
    preds = {"perfect": {c: clips[c].copy() for c in clips},
             "mean": {c: np.broadcast_to(mu, clips[c].shape).copy() for c in clips}}
    st, _, _ = _stats(clips, preds_map=preds)
    assert np.allclose(AG.r2(st, "perfect").per_channel, 1.0)
    assert np.allclose(AG.r2(st, "mean").per_channel, 0.0)


def test_train_mean_r2_is_never_stricter_than_class_mean_r2():
    """The correction recorded on 2026-08-12: the sample mean minimizes SSE on its own sample, so
    a TEST-derived denominator is SMALLER and the resulting R2 is LOWER. The 2026-08-11 claim that
    a TEST mean would inflate R2 had the direction backwards."""
    rng = np.random.default_rng(2)
    clips = {1: rng.normal(3.0, 1.0, size=(6, 2, N_CH)), 2: rng.normal(3.0, 1.0, size=(4, 2, N_CH))}
    preds = {"m": {c: clips[c] + rng.normal(0, 0.5, size=clips[c].shape) for c in clips}}
    st, _, _ = _stats(clips, preds_map=preds)
    train_mean = np.full(N_CH, 0.0)                                # deliberately off-centre
    r_class = AG.r2(st, "m", AG.CLASS_MEAN).per_channel
    r_train = AG.r2(st, "m", AG.TRAIN_MEAN, train_mean=train_mean).per_channel
    assert np.all(r_train >= r_class - 1e-12)
    assert np.all(r_train > r_class)                               # strict when the means differ


def test_ratio_of_means_survives_a_degenerate_clip_where_mean_of_ratios_explodes():
    """Clip 1 sits almost exactly on the class mean, so ITS OWN R2 denominator is ~0. Averaging
    per-clip R2 would be dominated by that clip's exploding ratio; aggregating SSE first is stable.
    This is the whole reason Q2 chose a ratio of means."""
    eps = 1e-6
    c1 = np.full((3, 1, N_CH), 1.0); c1[0, 0, :] += eps            # nearly constant...
    c2 = np.stack([np.full((1, N_CH), v) for v in (-4.0, 6.0, 1.0)])  # ...and its own mean (1.0)
    clips = {1: c1, 2: c2}                                         # equals clip 2's mean, hence
    #                                                                the class mean -> SST_1 ~ 0
    preds = {"m": {1: c1 + 0.5, 2: c2 + 0.5}}
    st, _, _ = _stats(clips, preds_map=preds)
    mu = AG.clip_balanced_mean(st)
    den = AG.constant_sse(st, mu)
    per_clip_r2 = 1.0 - (st.sse["m"][:, 0] / st.n_valid[:, 0]) / (den[:, 0] / st.n_valid[:, 0])
    assert abs(per_clip_r2[0]) > 1e4                               # degenerate clip explodes
    ours = AG.r2(st, "m").per_channel[0]
    assert np.isfinite(ours) and abs(ours) < 10                    # aggregate stays sane


def test_masked_channel_drops_only_that_channel_and_is_counted():
    """CoP is masked at low force, so a clip can be usable for F and unusable for CoP. The clip
    must drop out of the CoP channels ONLY, and n_clips must show it."""
    clips = {1: np.ones((2, 1, N_CH)), 2: np.full((2, 1, N_CH), 3.0)}
    m1 = np.ones((2, 1, N_CH), bool); m1[:, :, 1:] = False          # clip 1: no valid CoP at all
    m2 = np.ones((2, 1, N_CH), bool)
    preds = {"m": {1: np.ones((2, 1, N_CH)), 2: np.full((2, 1, N_CH), 3.0)}}
    st, _, _ = _stats(clips, mask_map={1: m1, 2: m2}, preds_map=preds)
    assert np.array_equal(st.n_valid[0], np.array([2, 0, 0]))
    res = AG.r2(st, "m")
    assert np.array_equal(res.n_clips, np.array([2, 1, 1]))
    # the CoP mean must come from clip 2 only, so it is 3.0 (not the 2.0 both-clip average)
    assert np.allclose(AG.clip_balanced_mean(st), [2.0, 3.0, 3.0])


def test_within_clip_baseline_is_each_clips_own_variance():
    clips = {1: np.array([[[0.0] * N_CH], [[2.0] * N_CH]]),          # own mean 1.0, SST = 2.0
             2: np.array([[[5.0] * N_CH], [[5.0] * N_CH]])}          # constant -> SST = 0
    st, _, _ = _stats(clips)
    sse = AG.within_clip_sse(st)
    assert np.allclose(sse[:, 0], [2.0, 0.0])
    assert np.all(sse >= 0.0)


def test_row_selection_supports_the_layer3_sensitivity_analysis():
    clips = {1: np.ones((2, 1, N_CH)), 7: np.full((2, 1, N_CH), 2.0), 9: np.zeros((2, 1, N_CH))}
    st, _, _ = _stats(clips)
    assert np.array_equal(st.rows_of([7, 9]), np.array([1, 2]))
    assert np.array_equal(st.drop([7]), np.array([0, 2]))
    assert np.array_equal(st.rows_of([1234]), np.array([], dtype=np.int64))   # absent id ignored
    # a resample may repeat rows; a repeated clip must count twice
    mu_rep = AG.clip_balanced_mean(st, rows=np.array([0, 0, 1]))
    assert np.allclose(mu_rep, (1.0 + 1.0 + 2.0) / 3)


def test_skill_and_delta_r2_wiring():
    # targets must actually vary, else the mean baseline has zero SSE and R2 is legitimately NaN
    clips = {1: np.arange(4.0).reshape(4, 1, 1) * np.ones((1, 1, N_CH)),
             2: (np.arange(4.0) + 1).reshape(4, 1, 1) * np.ones((1, 1, N_CH))}
    preds = {"good": {c: clips[c] + 1.0 for c in clips},
             "bad": {c: clips[c] + 2.0 for c in clips}}
    st, _, _ = _stats(clips, preds_map=preds)
    assert np.allclose(AG.skill(st, "good", "bad"), 1.0 - 1.0 / 4.0)
    a, b = AG.r2(st, "good"), AG.r2(st, "bad")
    assert np.allclose(AG.delta_r2(a, b), a.per_channel - b.per_channel)
    with pytest.raises(ValueError):
        AG.delta_r2(a, AG.r2(st, "bad", AG.CLIP_MEAN))              # mismatched denominators


def test_headline_never_replaces_the_per_channel_table():
    res = AG.R2Result(per_channel=np.array([0.9, 0.1, np.nan]), n_clips=np.array([3, 3, 0]),
                      channels=("F_R", "CoPx_R", "CoPy_R"), model="m", baseline=AG.CLASS_MEAN)
    assert np.isclose(res.headline, 0.5)                            # NaN channel excluded
    assert set(res.as_dict()) == {"F_R", "CoPx_R", "CoPy_R"}


# ======================================================================== bootstrap.py ==
def test_same_seed_reproduces_the_same_interval():
    f = lambda rows: np.array([float(rows.sum())])                  # noqa: E731
    a = BS.bootstrap_paired(f, 20, b=64, seed=7)
    b = BS.bootstrap_paired(f, 20, b=64, seed=7)
    c = BS.bootstrap_paired(f, 20, b=64, seed=8)
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, c.samples)
    assert a.provenance["numpy_version"] == np.__version__
    assert a.provenance["bit_generator"] == "PCG64" and a.provenance["design"] == "paired"


def test_paired_point_estimate_is_the_observed_sample_not_the_resample_mean():
    st_calls = []

    def f(rows):
        st_calls.append(rows)
        return np.array([float(np.mean(rows))])

    res = BS.bootstrap_paired(f, 10, b=5, seed=0)
    assert np.array_equal(st_calls[0], np.arange(10))               # first call = observed sample
    assert np.isclose(res.point[0], 4.5)
    assert res.samples.shape == (5, 1)


def test_paired_sharing_is_structural_and_tightens_a_correlated_difference():
    """Two models whose per-clip errors are strongly correlated. Paired resampling (one index set
    per iteration, both models scored on it) must give a much tighter interval on the DIFFERENCE
    than resampling the two models independently would. This is the whole justification for Q4's
    G1 design, and with this API the sharing cannot be broken by accident: stat_fn only ever
    receives ONE row array."""
    rng = np.random.default_rng(3)
    n = 60
    difficulty = rng.normal(0, 5.0, size=n)          # shared clip difficulty (the nuisance term)
    err_a = difficulty + rng.normal(0, 0.1, size=n)
    err_b = difficulty + 0.5 + rng.normal(0, 0.1, size=n)

    paired = BS.bootstrap_paired(lambda r: np.array([err_a[r].mean() - err_b[r].mean()]),
                                 n, b=800, seed=0)
    indep = BS.bootstrap_two_sample(lambda ra, rb: np.array([err_a[ra].mean() - err_b[rb].mean()]),
                                    n, n, b=800, seed=0)
    w_paired = float(paired.hi[0] - paired.lo[0])
    w_indep = float(indep.hi[0] - indep.lo[0])
    assert w_paired < 0.2 * w_indep
    assert paired.excludes_zero[0]                   # the real -0.5 gap is detected...
    assert not indep.excludes_zero[0]                # ...and would be missed unpaired
    assert abs(paired.bias[0]) < 0.05


def test_stratified_resample_preserves_every_stratum_count():
    strata = np.array(["holding"] * 30 + ["sliding"] * 5 + ["wiping"] * 2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        rows = BS.resample_rows(rng, len(strata), strata)
        assert rows.size == len(strata)
        drawn = strata[rows]
        for label, want in (("holding", 30), ("sliding", 5), ("wiping", 2)):
            assert int((drawn == label).sum()) == want
    # unstratified draws do NOT preserve composition (that is exactly the failure mode
    # stratification exists to prevent when one action dominates a class)
    plain = strata[BS.resample_rows(np.random.default_rng(1), len(strata))]
    assert int((plain == "holding").sum()) != 30


def test_stratified_resample_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        BS.resample_rows(np.random.default_rng(0), 5, np.array(["a", "b"]))


def test_two_sample_resamples_the_two_groups_independently():
    seen = []
    BS.bootstrap_two_sample(lambda ra, rb: (seen.append((ra.copy(), rb.copy())), np.array([0.0]))[1],
                            12, 12, b=20, seed=0)
    pairs = seen[1:]                                          # skip the observed-sample call
    assert any(not np.array_equal(ra, rb) for ra, rb in pairs)
    assert all(ra.size == 12 and rb.size == 12 for ra, rb in pairs)


def test_percentile_ci_is_nan_tolerant():
    s = np.array([[1.0], [2.0], [np.nan], [3.0], [4.0]])
    lo, hi = BS.percentile_ci(s, alpha=0.5)
    assert np.isfinite(lo[0]) and np.isfinite(hi[0]) and lo[0] < hi[0]


def test_r2_bootstrap_recomputes_the_mean_inside_each_resample():
    """The class mean is part of the statistic, so a resample must use ITS OWN mean. Wiring it
    through `rows` (rather than freezing the full-sample mean) is what makes that true, and the
    sufficient statistics make it free."""
    rng = np.random.default_rng(4)
    clips = {i: rng.normal(size=(3, 2, N_CH)) for i in range(12)}
    preds = {"m": {i: clips[i] + rng.normal(0, 0.4, size=clips[i].shape) for i in clips}}
    st, _, _ = _stats(clips, preds_map=preds)
    res = BS.bootstrap_paired(lambda rows: AG.r2(st, "m", rows=rows).per_channel,
                              st.n_clips, b=200, seed=0)
    assert res.samples.shape == (200, N_CH)
    assert np.all(res.lo <= res.point + 1e-9) and np.all(res.point <= res.hi + 1e-9)
    # a resample's own mean differs from the full-sample mean -> the samples are not degenerate
    assert np.nanstd(res.samples[:, 0]) > 0


# NOTE: the GRU-aggregate fork is tested in tests/test_opentouch_gru_aggregate.py, which is
# skipped wholesale when torch is unavailable. It is a SEPARATE file on purpose: a module-level
# importorskip here would also skip every trait/aggregate/bootstrap test above, which are pure
# numpy and must always run.
