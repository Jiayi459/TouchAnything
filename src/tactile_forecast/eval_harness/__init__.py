"""Frozen evaluation harness for tactile forecasting (aggregate force + CoP).

This package defines — in exactly one place — the target, splits, masking, and metrics
that EVERY future tactile-forecasting model is scored against, plus three classical
baselines (persistence, seasonal-naive, AR). It is intended to be FROZEN: import these
definitions everywhere; do not re-implement them per-model.

TIME-INDEXING CONVENTION (used throughout this package):
    Every prediction is indexed by TARGET time, i.e. the time the predicted value refers
    to. A forecast ISSUED at origin time t for horizon h (h = 1..H) refers to target time
    t+h and is stored/compared at index t+h, carrying metadata (t, h). Baselines only ever
    read observations at times <= t (strict causality; see the causality unit test).

CAUSALITY: no non-causal / zero-phase operations anywhere in this package. There is no
filtfilt. If any filtering is added, it MUST be causal (scipy lfilter/sosfilt) and carry
an explicit comment at the call site (see masking.py / baselines for the pattern).
"""
# Definitional modules only. `evaluate` is the entry point (run via -m); importing it here
# would double-import under `python -m ...evaluate`, so it is intentionally left out.
from . import config, splits, dataset, masking, metrics  # noqa: F401
