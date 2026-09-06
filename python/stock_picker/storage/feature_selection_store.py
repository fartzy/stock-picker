"""Repository for the current per-run feature selection: which features a
training run should be restricted to, distinct from `PrunedFeatureStore`'s
permanent block-list. Pruning is a quality judgment ("this feature is bad,
exclude it everywhere"); selection is an experiment ("for the next run, only
use these") -- e.g. trying a new feature in isolation or A/B-ing a subset.

`None` means "no explicit selection" -- every feature not pruned. An empty
set is a real (if unusual) selection of zero features, not the same as "no
selection," so this is stored as `{"included_features": [...] | null}`
rather than an empty list standing in for both.
"""

from __future__ import annotations

import json
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "feature_selection"


class FeatureSelectionStore:
    """Reads and writes the single current feature selection."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "selection.json"

    def read(self) -> set[str] | None:
        if not self._path.exists():
            return None
        included = json.loads(self._path.read_text()).get("included_features")
        return set(included) if included is not None else None

    def write(self, included_features: set[str] | None) -> None:
        included = sorted(included_features) if included_features is not None else None
        self._path.write_text(json.dumps({"included_features": included}, indent=2))

    def clear(self) -> None:
        self.write(None)
