"""Append-only log of completed/failed training runs -- what let "click into a
run" answer "what tickers/dates/features fed this, and how long did it take,"
not just the latest job's status.

Mutation semantics follow trade_store.py's append-only log (no edit/remove --
a run record never changes once written), not feature_exclusion_store.py's
mutate-with-removal shape (that one tracks a *current* pruned set; this
tracks a history of *past events*). Physical format stays JSON rather than
trade_store.py's Parquet, though: a run record is nested/variable-shape
(model_specs, fold_metrics, threshold_sweep are each lists of dicts), not the
flat scalar-column schema Parquet suits -- same reasoning
training_config_store.py/feature_exclusion_store.py already use for their own
list-of-dict records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "training_runs"


@dataclass
class TrainingRunRecord:
    run_id: str
    status: Literal["completed", "failed"]
    started_at: str
    completed_at: str
    duration_seconds: float
    git_commit: str | None = None
    # None on a run that failed before this provenance was known -- see
    # training/main.py's run_training() for why a failure can't always
    # recover these.
    train_tickers: list[str] | None = None
    holdout_tickers: list[str] | None = None
    date_range: tuple[str, str] | None = None
    resolved_features: list[str] | None = None
    model_specs: list[dict] | None = None
    fold_metrics: list[dict] | None = None
    holdout_metrics: dict | None = None
    threshold_sweep: list[dict] | None = None
    error: str | None = None


class TrainingRunStore:
    """Appends and reads the training-run history as a single JSON file."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "runs.json"

    def _load(self) -> list[dict]:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return []

    def append(self, record: TrainingRunRecord) -> None:
        records = self._load()
        records.append(asdict(record))
        self._path.write_text(json.dumps(records, indent=2))

    def read_all(self) -> list[TrainingRunRecord]:
        """Every recorded run, newest first."""
        records = sorted(self._load(), key=lambda entry: entry["started_at"], reverse=True)
        return [
            TrainingRunRecord(
                **{
                    **record,
                    # JSON has no tuple type -- round-trips as a list, so
                    # restore the dataclass's declared (min, max) shape.
                    "date_range": tuple(record["date_range"]) if record.get("date_range") else None,
                }
            )
            for record in records
        ]
