"""Read-only wiring for training/eval to see excluded feature columns.

Mutations (prune/unprune) go straight through PrunedFeatureStore from
api/routes.py. New research candidates are withheld separately until a
walk-forward result merits promotion; they are not labeled low-quality prunes.
"""

from __future__ import annotations

from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.features.breadth import BREADTH_COLUMNS
from stock_picker.features.trajectory import PATH_COLUMNS
from stock_picker.features.volatility import SEMIVOL_WINDOWS, SHORT_VOLATILITY_LAGS

RESEARCH_CANDIDATE_COLUMNS = frozenset({
    "volatility_32d",
    "absolute_return_2d",
    *(f"volatility_5d_lag_{lag}d" for lag in SHORT_VOLATILITY_LAGS),
    *(f"{side}_semivol_{window}d" for side in ("upside", "downside") for window in SEMIVOL_WINDOWS),
    *(f"semivol_balance_{window}d" for window in SEMIVOL_WINDOWS),
    *BREADTH_COLUMNS,
    *PATH_COLUMNS,
})


def pruned_features() -> set[str]:
    return PrunedFeatureStore().read() | RESEARCH_CANDIDATE_COLUMNS
