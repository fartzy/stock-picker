"""Repository for pruned (excluded-from-training) feature names.

Unlike trade_store.py's append-only log, un-pruning must remove a row, so
this supports idempotent add/remove against a single Parquet file --
matching universe_store.py's read-modify-write shape.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "pruned_features"

_COLUMNS = ["feature", "pruned_at"]


class PrunedFeatureStore:
    """Reads and mutates the set of pruned (training-excluded) feature names."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "pruned.parquet"

    def _load(self) -> pd.DataFrame:
        if self._path.exists():
            return pd.read_parquet(self._path)
        return pd.DataFrame(columns=_COLUMNS)

    def read(self) -> set[str]:
        return set(self._load()["feature"])

    def prune(self, feature: str) -> None:
        pruned = self._load()
        if feature in set(pruned["feature"]):
            return
        new_row = pd.DataFrame([{"feature": feature, "pruned_at": datetime.now().astimezone().isoformat()}])
        pruned = new_row if pruned.empty else pd.concat([pruned, new_row], ignore_index=True)
        pruned.to_parquet(self._path, index=False)

    def unprune(self, feature: str) -> None:
        pruned = self._load()
        remaining = pruned[pruned["feature"] != feature]
        remaining.to_parquet(self._path, index=False)
