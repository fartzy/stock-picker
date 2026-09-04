"""Repository for persisting computed per-ticker feature tables to local Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "features"


class FeatureStore:
    """Reads and writes per-ticker feature tables as Parquet files."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ticker: str) -> Path:
        return self._data_dir / f"{ticker}.parquet"

    def write(self, ticker: str, features: pd.DataFrame) -> None:
        features.to_parquet(self._path_for(ticker))

    def read(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self._path_for(ticker))
