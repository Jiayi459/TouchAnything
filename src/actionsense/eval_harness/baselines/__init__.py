"""Classical baselines any learned model must beat. All are strictly CAUSAL:
`predict(hist, H)` reads only `hist` (observations up to and including origin t) and
returns (H, 6) indexed by target time t+h. See base.Baseline for the contract.
"""
from .base import Baseline, predict_series, origins  # noqa: F401
from .persistence import Persistence                # noqa: F401
from .seasonal import SeasonalNaive                 # noqa: F401
from .ar import AR                                  # noqa: F401

ALL = [Persistence, SeasonalNaive, AR]
