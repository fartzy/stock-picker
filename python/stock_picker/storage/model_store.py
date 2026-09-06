"""Repository for persisting trained model ensembles to local files.

Plain pickle rather than each library's native format: an `Ensemble`
(training/ensemble.py) can hold a mix of LightGBM boosters and scikit-learn
estimators, and pickle handles both uniformly with zero per-type serializer
code, at the cost of being less portable across library versions than a
native format -- an acceptable tradeoff for locally-retrained models.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "models"


class ModelStore:
    """Reads and writes named, pickled model ensembles."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str) -> Path:
        return self._data_dir / f"{name}.pkl"

    def write(self, name: str, model: Any) -> None:
        with open(self._path_for(name), "wb") as f:
            pickle.dump(model, f)

    def read(self, name: str) -> Any:
        with open(self._path_for(name), "rb") as f:
            return pickle.load(f)

    def exists(self, name: str) -> bool:
        return self._path_for(name).exists()
