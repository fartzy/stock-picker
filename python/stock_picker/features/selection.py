"""Read-only wiring for training to see the current per-run feature selection.

Mutations (set/clear) go straight through TrainingConfigStore from
api/routes.py -- there's no transform to share, just a store method call.
"""

from __future__ import annotations

from stock_picker.storage.training_config_store import TrainingConfigStore


def selected_features() -> set[str] | None:
    included = TrainingConfigStore().read().included_features
    return set(included) if included is not None else None
