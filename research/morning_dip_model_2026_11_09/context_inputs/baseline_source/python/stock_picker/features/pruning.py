"""Read-only wiring for training/eval to see excluded feature columns.

Mutations (prune/unprune) go straight through PrunedFeatureStore from
api/routes.py. Every persisted feature follows the same explicit prune list.
"""

from __future__ import annotations

from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.features.breadth import BREADTH_COLUMNS
from stock_picker.features.trajectory import PATH_COLUMNS
from stock_picker.features.volatility import SEMIVOL_WINDOWS, SHORT_VOLATILITY_LAGS

PRICE_PATTERN_COLUMNS = frozenset({
    "volatility_32d",
    "absolute_return_2d",
    *(f"volatility_5d_lag_{lag}d" for lag in SHORT_VOLATILITY_LAGS),
    *(f"{side}_semivol_{window}d" for side in ("upside", "downside") for window in SEMIVOL_WINDOWS),
    *(f"semivol_balance_{window}d" for window in SEMIVOL_WINDOWS),
    *BREADTH_COLUMNS,
    *PATH_COLUMNS,
})

# Historical experiment code uses this group name for paired ablations.
# It is metadata only and does not exclude these inputs from training.
RESEARCH_CANDIDATE_COLUMNS = PRICE_PATTERN_COLUMNS


def pruned_features() -> set[str]:
    return PrunedFeatureStore().read()
