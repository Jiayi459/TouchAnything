"""TRAIT CLASS (smooth vs abrupt) — the frozen, pre-registered definition for G2.

THIS FILE IS A PRE-REGISTRATION ARTIFACT. It is committed BEFORE any OpenTouch TRAIN/TEST
split exists and BEFORE any G2 number has been computed, so its git timestamp is evidence
that the class definition was not chosen to produce a result. Nothing in here may be edited
in response to a measured outcome (see "HARD DISCIPLINE" below).

Replaces the ad-hoc `SMOOTH = {...}` set that lived in a throwaway script on 2026-08-07 and
was never reviewed. That set is superseded, not merely relocated: applying the rubric below
to the full action vocabulary moves `cutting`/`flipping`/`scooping`/`serving`/`eating`/
`drinking` OUT of smooth and `holding`/`sliding` IN (SESSION_LOG 2026-08-12).


LAYER 1 — THE RUBRIC (a priori, physical, sensor-agnostic)
==========================================================
The unit of classification is the ACTION type, not the clip (see "GRANULARITY" below).

An action is ABRUPT iff its typical execution contains DISCRETE CONTACT-STATE TRANSITION
EVENTS — impact (collision), grasp/release (the instant of gripping or letting go), or a
sudden change of contact surface — and such events are EXPECTED to occur within the action's
typical time window.

An action is SMOOTH iff its typical execution is SUSTAINED CONTACT + CONTINUOUS FORCE
MODULATION: hand (or held tool) stays in contact with the object and the force varies through
continuous muscular regulation, with no expected impact event.

Two refinements were needed to apply this consistently; both are recorded here because they
carry real classification weight and a reviewer is entitled to see them stated in advance:

  (R1) EVENTS ARE COUNTED AT THE HAND'S MECHANICAL COUPLING, INCLUDING THROUGH A HELD TOOL.
       The glove measures the hand. A knife striking the cutting board does not change the
       hand-knife contact state, but the impulse is transmitted rigidly to the hand and is
       plainly visible in F. So tool-environment collisions COUNT as impact events. (This is
       what forces `cutting` -> abrupt, and it is the same reasoning that makes `scooping`
       (spoon striking the bowl) and `serving` suspect.)

  (R2) CONSTITUTIVE EVENTS COUNT; INCIDENTAL ONSET/OFFSET FRAMING A SUSTAINED PHASE DOES NOT.
       Almost every action begins by establishing contact, so a bare "contact onset" cannot
       be the discriminator or every action would be abrupt. The question is whether the
       discrete event IS the action (remove it and the action did not happen: you cannot
       `press` without the actuation transition, cannot `place` without release) or whether
       it merely precedes/follows a sustained phase that constitutes the action (you grasp a
       jug before `pouring`, but pouring is the sustained tilt; the grasp is preparatory and
       typically falls outside or at the margin of the annotated window). Without (R2),
       `pouring` and `carrying` would be abrupt, which contradicts the physics of their force
       signals and the user's explicit ruling.


LAYER 3 — THE PRE-REGISTERED CONTENTIOUS SUBSET (defined now, before any result)
================================================================================
An action is CONTENTIOUS iff the rubric's verdict VARIES ACROSS TYPICAL INSTANCES of that
action — i.e. the indeterminacy is in the action type, not in the classifier's confidence.
`turning` a knob is sustained rotation; `turning` a page ends in release. Both are "turning".

Every G2 primary result is reported WITH a secondary analysis that drops CONTENTIOUS and
recomputes R2/dR2. If the direction of the conclusion is unchanged, the conclusion does not
depend on how the boundary actions were assigned. This costs one extra aggregation pass and
no model refitting, because per-clip SSE is materialized as an explicit intermediate product
(see aggregate.py).


HARD DISCIPLINE
===============
The Layer-2 manipulation check (below) VALIDATES this classification; it never revises it.
If a measured action lands clearly on the "wrong" side of the signal statistic, that goes in
the limitations section. Relabelling on the basis of the check would make the definition
post-hoc and would destroy the whole point of committing this file early.


GRANULARITY (a known, deliberate limitation)
============================================
Classification is per ACTION, which assumes all clips of one action are homogeneous. In
reality execution style varies by participant (one person chops, another saws gently), so an
action's clips may straddle the boundary. `per_action_stat` reports the WITHIN-action spread
precisely so this can be quantified and discussed. It still does not license per-CLIP
classification: selecting clips by a signal property and then measuring the predictability of
that signal is a strictly worse circularity than the one it would be trying to fix.


VOCABULARY COMPLETENESS
=======================
OpenTouch has 66 distinct action strings (measured 2026-08-10). 30 are audited below: the 14
with n>=30 plus the 16 of the superseded 2026-08-07 set. The remaining ~36 are long-tail
(each n<32) and their exact strings are not known until the labels are in hand. `trait_class`
RAISES on an unaudited action rather than defaulting it to abrupt — a silent default would
quietly fill the abrupt class with whatever the tail happens to contain. `unaudited()` +
`partition()` make the post-download completeness pass explicit and countable.
"""
from __future__ import annotations

import collections

import numpy as np

SMOOTH = "smooth"
ABRUPT = "abrupt"

# ---------------------------------------------------------------------------------------
# THE AUDIT TABLE. Provenance per entry:
#   [U]  the user's explicit ruling, 2026-08-12 (verbatim, not re-derived)
#   [R]  derived by applying the rubric above; subject to the same discipline once committed
# `+`  = also in CONTENTIOUS (Layer 3)
# ---------------------------------------------------------------------------------------
TRAIT_CLASS: dict[str, str] = {
    # -- SMOOTH: sustained contact, continuous force modulation, no expected discrete event --
    "holding":    SMOOTH,   # [R] contact held, force modulated continuously; zero constitutive events.
    "sliding":    SMOOTH,   # [R] structurally identical to wiping: sustained contact + continuous force.
    "wiping":     SMOOTH,   # [R] unchanged from the superseded set.
    "cleaning":   SMOOTH,   # [R] wiping-class.
    "scraping":   SMOOTH,   # [U] "持续接触、持续力调制,虽然力可能大但没有离散事件 → smooth(和 wiping 同构)".
    "pouring":    SMOOTH,   # [U] continuous tilt; load changes continuously as liquid leaves.
    "stirring":   SMOOTH,   # [U] ruled smooth explicitly ("无争议"), overriding a possible
                            #     spoon-vs-pot-wall reading under (R1).
    "spreading":  SMOOTH,   # [U]
    "drawing":    SMOOTH,   # [U]
    "writing":    SMOOTH,   # [U]
    "carrying":   SMOOTH,   # [U] sustained grasp under continuous load; grasp is preparatory (R2).
    "lowering":   SMOOTH,   # [R]+ controlled descent under load is the action; the terminal surface
                            #      contact is at the window margin (R2) but often inside it -> contentious.

    # -- ABRUPT: discrete, constitutive contact-state transitions --
    "picking up":  ABRUPT,  # [R] grasp + lift-off ARE the action (R2).
    "placing":     ABRUPT,  # [R] surface contact + release ARE the action.
    "pressing":    ABRUPT,  # [R] the actuation transition IS the action.
    "grasping":    ABRUPT,  # [R] the grasp instant IS the action.
    "touching":    ABRUPT,  # [R] contact establishment IS the action.
    "removing":    ABRUPT,  # [R] detachment IS the action.
    "pulling":     ABRUPT,  # [R]+ displacement is sustained, but instances typically engage an
                            #      object and hit a terminal stop; varies by instance -> contentious.
    "pushing":     ABRUPT,  # [R]+ same reasoning as pulling.
    "adjusting":   ABRUPT,  # [R]+ re-grips (release+regrasp) in some instances, sustained in others.
    "turning":     ABRUPT,  # [R]+ knob = sustained torque; page/cap = release. Varies.
    "moving":      ABRUPT,  # [R]+ slide-with-contact vs pick-and-place. Varies.
    "inspecting":  ABRUPT,  # [R]+ usually acquires the object (grasp), sometimes pure sustained hold.
    "cutting":     ABRUPT,  # [U]+ "刀每次下压到砧板是一次 impact,锯切动作里刀刃反复脱离/重新咬合材料".
    "flipping":    ABRUPT,  # [U]+ "铲子插入、食物离手(release)、落回接住(impact)".
    "scooping":    ABRUPT,  # [U]+ user-flagged: utensil-container collisions (R1).
    "serving":     ABRUPT,  # [U]+ user-flagged: scoop + release.
    "eating":      ABRUPT,  # [U]+ user-flagged: "'送入口中'环节可能含 release" + utensil-plate contacts.
    "drinking":    ABRUPT,  # [U]+ user-flagged; hand-cup coupling is sustained, but set-down
                            #      (impact + release) is expected inside a 2.8 s median window.
}

CONTENTIOUS: frozenset[str] = frozenset({
    "lowering", "pulling", "pushing", "adjusting", "turning", "moving", "inspecting",
    "cutting", "flipping", "scooping", "serving", "eating", "drinking",
})

assert CONTENTIOUS <= set(TRAIT_CLASS), "CONTENTIOUS must only name audited actions"


def normalize_action(action: str | None) -> str:
    """Manifest `action` -> canonical key. Lowercased, whitespace-collapsed. Empty/None ->
    "" (missing label), which is NOT a class and must be dropped explicitly by the caller."""
    return " ".join((action or "").split()).lower()


class UnauditedAction(KeyError):
    """Raised for an action string the rubric has not been applied to. Deliberately fatal:
    defaulting the long tail into `abrupt` would silently define the majority class."""


def trait_class(action: str | None) -> str:
    """-> "smooth" | "abrupt". Raises UnauditedAction for unknown/missing actions."""
    key = normalize_action(action)
    if not key:
        raise UnauditedAction("empty action label (missing annotation) has no trait class")
    if key not in TRAIT_CLASS:
        raise UnauditedAction(
            f"action {key!r} has not been audited against the trait rubric; apply the rubric "
            f"in src/opentouch/trait.py and commit the verdict BEFORE scoring it"
        )
    return TRAIT_CLASS[key]


def is_contentious(action: str | None) -> bool:
    """True iff this action is in the pre-registered contentious subset (Layer 3)."""
    return normalize_action(action) in CONTENTIOUS


def unaudited(actions) -> set[str]:
    """Canonical action strings present in `actions` that the table does not cover (empty
    labels excluded -- they are reported separately by `partition`)."""
    seen = {normalize_action(a) for a in actions}
    return {a for a in seen if a and a not in TRAIT_CLASS}


def partition(actions: dict[int, str]) -> dict[str, list[int]]:
    """clip idx -> action  ==>  {"smooth", "abrupt", "unlabeled", "unaudited"} -> [idx].

    Every input idx lands in exactly one bucket, so `unlabeled` and `unaudited` are COUNTABLE
    drops that must be reported alongside any G2 number rather than vanishing.
    """
    out: dict[str, list[int]] = {SMOOTH: [], ABRUPT: [], "unlabeled": [], "unaudited": []}
    for i, a in sorted(actions.items()):
        key = normalize_action(a)
        if not key:
            out["unlabeled"].append(i)
        elif key not in TRAIT_CLASS:
            out["unaudited"].append(i)
        else:
            out[TRAIT_CLASS[key]].append(i)
    return out


def contentious_ids(actions: dict[int, str]) -> list[int]:
    """Clip idxs whose action is in CONTENTIOUS -- the set the Layer-3 sensitivity analysis
    DROPS before recomputing R2/dR2."""
    return sorted(i for i, a in actions.items() if is_contentious(a))


# =======================================================================================
# LAYER 2 — MANIPULATION CHECK (statistics FROZEN here; executed only after splits exist)
# =======================================================================================
# Purpose: show that the two semantically-defined classes are separated at the SIGNAL level.
# This is a descriptive validation, reported as a figure. It is NOT an input to the labels.
#
# Statistic: per clip, the 95th percentile of |dF| (absolute first difference of the causally
# filtered total-force channel) -- i.e. how large the large frame-to-frame force jumps are.
# Aggregated to a per-action MEDIAN over that action's clips. A frequency-domain corroboration
# (fraction of force power above HF_CUTOFF_HZ) is reported alongside.
#
# WHERE IT MAY BE COMPUTED: the OpenTouch TRAIN split ONLY. Same inheritance discipline as
# Norm and force_thresholds -- a sensor-dependent quantity is frozen as a RECOMPUTABLE RULE
# and applied to a new dataset's train split, never fitted on data that will be scored.
#
# RATE: taken from cfg (fps_raw / downsample = 30 Hz for OpenTouch), never hardcoded. NOTE:
# the 2026-08-12 instruction said "15 Hz"; the frozen OpenTouch config is 30 Hz native with
# downsample: 1, so |dF| here is a 33 ms jump, not 67 ms (SESSION_LOG 2026-08-12).
DELTA_F_PERCENTILE = 95.0
HF_CUTOFF_HZ = 3.0
MIN_DIFFS = 20              # clips yielding fewer first-differences are excluded from the check
CAUSAL_SMOOTH_FRAMES = 3    # 100 ms at 30 Hz; k=1 (no filter) reported alongside as a check
                            # that the conclusion does not hinge on this choice.


def causal_smooth(x: np.ndarray, k: int = CAUSAL_SMOOTH_FRAMES) -> np.ndarray:
    """Strictly causal moving average of length k over a 1-D signal: output[t] uses only
    x[t-k+1..t], with the head left-padded by x[0] (no future leakage, no length change).
    k=1 is the identity. Suppresses single-frame sensor noise being read as a contact event."""
    if k <= 1:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    padded = np.concatenate([np.full(k - 1, x[0]), x])
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def delta_f_p95(force: np.ndarray, k: int = CAUSAL_SMOOTH_FRAMES,
                pct: float = DELTA_F_PERCENTILE, min_diffs: int = MIN_DIFFS) -> float | None:
    """Per-clip statistic: `pct`th percentile of |diff(causal_smooth(force))|.
    -> None if the clip is too short to estimate a high percentile (excluded, not zero-filled)."""
    force = np.asarray(force, dtype=np.float64).ravel()
    if force.size - 1 < min_diffs:
        return None
    return float(np.percentile(np.abs(np.diff(causal_smooth(force, k))), pct))


def hf_energy_fraction(force: np.ndarray, fps: float, cutoff_hz: float = HF_CUTOFF_HZ,
                       min_diffs: int = MIN_DIFFS) -> float | None:
    """Corroborating frequency-domain statistic: fraction of the mean-removed force's power
    spectrum above `cutoff_hz`. Descriptive only (an FFT over the whole clip is not causal —
    acceptable here because this quantity never enters a forecast, only a validation figure).
    -> None if the clip is too short."""
    force = np.asarray(force, dtype=np.float64).ravel()
    if force.size - 1 < min_diffs:
        return None
    x = force - force.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fps)
    total = power[1:].sum()                      # drop DC (removed by construction anyway)
    if total <= 0:
        return None
    return float(power[1:][freqs[1:] > cutoff_hz].sum() / total)


def per_action_stat(values: dict[int, float | None], actions: dict[int, str]
                    ) -> dict[str, dict[str, float | int]]:
    """Aggregate a per-clip statistic to per-action median + WITHIN-action spread.

    `values[idx]` may be None (clip excluded, e.g. too short); those clips are dropped and
    counted. The IQR/min/max columns exist for the granularity limitation discussed above:
    an action whose clips straddle the two classes shows up here as a wide within-action
    spread, which is reported, not acted upon.
    """
    by_action: dict[str, list[float]] = collections.defaultdict(list)
    dropped: collections.Counter = collections.Counter()
    for i, v in values.items():
        key = normalize_action(actions.get(i))
        if v is None:
            dropped[key] += 1
        else:
            by_action[key].append(float(v))
    out: dict[str, dict[str, float | int]] = {}
    for key, vals in sorted(by_action.items()):
        a = np.asarray(vals)
        out[key] = {"median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
                    "p75": float(np.percentile(a, 75)), "min": float(a.min()),
                    "max": float(a.max()), "n_clips": int(a.size),
                    "n_dropped": int(dropped.get(key, 0)),
                    "trait": TRAIT_CLASS.get(key, "UNAUDITED"),
                    "contentious": key in CONTENTIOUS}
    return out
