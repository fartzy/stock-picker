"""Repository for the current per-run training configuration: which features
and which ensemble model types the next training run should use. Replaces the
former FeatureSelectionStore -- both answer the same underlying question
("what should the next training run look like"), so one JSON file holds both
rather than two parallel small stores. Still no database: this is a local
single-user tool, and a single JSON blob is easier to reason about than
managing schema/migrations for something this small.

Distinct from PrunedFeatureStore -- pruning is a permanent quality judgment
("this feature is bad, exclude everywhere"); this is a per-run experiment
("for the next run, use these features/models"). `included_features=None`
and `model_choices=None` both mean "no explicit choice, use the code's own
defaults" -- an empty list is a real (if unusual) explicit choice of zero,
so it's stored distinctly from null, same reasoning FeatureSelectionStore
used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "training_config"


@dataclass
class ModelChoice:
    model_type: str
    weight: float = 1.0


@dataclass
class TrainingConfig:
    included_features: list[str] | None = None
    model_choices: list[ModelChoice] | None = None


class TrainingConfigStore:
    """Reads/writes the single current training configuration. Each half
    (features, model choices) is read-modify-written independently so the UI
    can change one without disturbing the other."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "config.json"

    def read(self) -> TrainingConfig:
        if not self._path.exists():
            return TrainingConfig()
        raw = json.loads(self._path.read_text())
        model_choices = raw.get("model_choices")
        return TrainingConfig(
            included_features=raw.get("included_features"),
            model_choices=(
                [ModelChoice(**choice) for choice in model_choices] if model_choices is not None else None
            ),
        )

    def write_included_features(self, included_features: set[str] | None) -> None:
        config = self.read()
        config.included_features = sorted(included_features) if included_features is not None else None
        self._save(config)

    def write_model_choices(self, model_choices: list[ModelChoice] | None) -> None:
        config = self.read()
        config.model_choices = model_choices
        self._save(config)

    def _save(self, config: TrainingConfig) -> None:
        self._path.write_text(
            json.dumps(
                {
                    "included_features": config.included_features,
                    "model_choices": (
                        [{"model_type": c.model_type, "weight": c.weight} for c in config.model_choices]
                        if config.model_choices is not None
                        else None
                    ),
                },
                indent=2,
            )
        )
