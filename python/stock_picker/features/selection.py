"""Read-only wiring for training to see the current per-run feature selection.

Mutations (set/clear) go straight through FeatureSelectionStore from
api/routes.py -- there's no transform to share, just a store method call.
"""

from __future__ import annotations

from stock_picker.storage.feature_selection_store import FeatureSelectionStore


def selected_features() -> set[str] | None:
    return FeatureSelectionStore().read()
