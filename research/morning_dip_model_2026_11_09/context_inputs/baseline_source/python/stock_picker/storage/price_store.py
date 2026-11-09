"""Repository for persisting OHLCV price history to local Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "prices"
OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def sanitize_price_history(history: pd.DataFrame) -> pd.DataFrame:
    """Keep a bar's known fields, but mark impossible OHLC prices as missing.

    A provider can supply a zero open alongside a valid close. Keeping the bar
    preserves close-based history without inventing an opening trade or an
    infinite open-to-close return.
    """
    cleaned = history.copy()
    for column in OHLC_COLUMNS:
        if column not in cleaned:
            continue
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = values.where(values.gt(0) & values.lt(float("inf")))
    return cleaned


class PriceStore:
    """Reads and writes per-ticker OHLCV history as Parquet files."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ticker: str) -> Path:
        return self._data_dir / f"{ticker}.parquet"

    def write(self, ticker: str, history: pd.DataFrame) -> None:
        sanitize_price_history(history).to_parquet(self._path_for(ticker))

    def read(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self._path_for(ticker))
