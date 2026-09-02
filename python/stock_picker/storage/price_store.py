"""Repository for persisting OHLCV price history to local Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/prices")


class PriceStore:
    """Reads and writes per-ticker OHLCV history as Parquet files."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ticker: str) -> Path:
        return self._data_dir / f"{ticker}.parquet"

    def write(self, ticker: str, history: pd.DataFrame) -> None:
        history.to_parquet(self._path_for(ticker))

    def read(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self._path_for(ticker))
