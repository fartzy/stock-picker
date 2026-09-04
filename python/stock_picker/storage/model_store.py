"""Repository for persisting trained LightGBM models to local files."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "models"


class ModelStore:
    """Reads and writes named LightGBM boosters using their native text format."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str) -> Path:
        return self._data_dir / f"{name}.txt"

    def write(self, name: str, model: lgb.Booster) -> None:
        model.save_model(str(self._path_for(name)))

    def read(self, name: str) -> lgb.Booster:
        return lgb.Booster(model_file=str(self._path_for(name)))
