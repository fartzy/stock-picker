"""Repository for pruned (excluded-from-training) feature names, with a reason
per pruned feature (e.g. "high correlation to return_3d (r=0.996)").

Unlike trade_store.py's append-only log, un-pruning must remove an entry, so this
supports idempotent add/remove -- matching universe_store.py's read-modify-write
shape. Plain JSON, not Parquet: this is a handful of small human-readable records,
not a columnar dataset -- no sqlite/duckdb needed at this size.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "pruned_features"
DEFAULT_REASON = "manually pruned"


class PrunedFeatureStore:
    """Reads and mutates the archive of pruned (training-excluded) features."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "pruned.json"

    def _load(self) -> list[dict]:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return []

    def _save(self, entries: list[dict]) -> None:
        self._path.write_text(json.dumps(entries, indent=2))

    def read(self) -> set[str]:
        return {entry["feature"] for entry in self._load()}

    def read_all(self) -> list[dict]:
        """Full archive detail (feature, reason, pruned_at), newest first."""
        return sorted(self._load(), key=lambda entry: entry["pruned_at"], reverse=True)

    def prune(self, feature: str, reason: str = DEFAULT_REASON) -> None:
        entries = self._load()
        if any(entry["feature"] == feature for entry in entries):
            return
        entries.append(
            {
                "feature": feature,
                "reason": reason,
                "pruned_at": datetime.now().astimezone().isoformat(),
            }
        )
        self._save(entries)

    def unprune(self, feature: str) -> None:
        entries = [entry for entry in self._load() if entry["feature"] != feature]
        self._save(entries)
