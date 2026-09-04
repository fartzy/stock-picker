"""Read-only wiring for training/eval to see the current pruned-feature set.

Mutations (prune/unprune) go straight through PrunedFeatureStore from
api/routes.py -- there's no transform to share, just a store method call.
"""

from __future__ import annotations

from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore


def pruned_features() -> set[str]:
    return PrunedFeatureStore().read()
