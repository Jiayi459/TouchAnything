"""Classical baselines any learned model must beat. All are strictly CAUSAL.

Persistence is imported UNCHANGED from src.actionsense.eval_harness.baselines.persistence:
it has no hardcoded channel count (`np.repeat` shape follows the input) and no
ActionSense-specific logic, so it needs no fork (see ../masking.py's docstring for the
fork/import policy). SeasonalNaive, AR, Baseline/predict_series/origins ARE forks (they
had hardcoded 6); see their own docstrings.
"""
from src.actionsense.eval_harness.baselines.persistence import Persistence  # noqa: F401  (unchanged, reused)

from .base import Baseline, predict_series, origins  # noqa: F401
from .seasonal import SeasonalNaive                 # noqa: F401
from .ar import AR                                  # noqa: F401

ALL = [Persistence, SeasonalNaive, AR]
