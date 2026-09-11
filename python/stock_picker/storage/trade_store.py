"""Repository for logged trade executions.

Mirrors universe_store.py's pattern: an append-only log, read-modify-write
against a Parquet file (trade volume here is a handful of rows per day, so
a full rewrite per append is fine). Also writes trades.csv next to it --
same columns, human-readable, always in sync.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "trades"

_COLUMNS = ["ticker", "side", "shares", "price", "executed_at"]


@dataclass
class Trade:
    ticker: str
    side: str  # "buy" | "sell"
    shares: float
    price: float
    executed_at: str  # ISO 8601 with UTC offset, e.g. datetime.now().astimezone().isoformat()


class TradeStore:
    """Reads and appends the trade log as a Parquet file."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "trades.parquet"
        self._csv_path = self._data_dir / "trades.csv"

    def _load(self) -> pd.DataFrame:
        if self._path.exists():
            return pd.read_parquet(self._path)
        return pd.DataFrame(columns=_COLUMNS)

    def _write(self, trades: pd.DataFrame) -> None:
        trades.to_parquet(self._path, index=False)
        trades.to_csv(self._csv_path, index=False)

    def append(self, trade: Trade) -> None:
        trades = self._load()
        new_row = pd.DataFrame([asdict(trade)])
        trades = new_row if trades.empty else pd.concat([trades, new_row], ignore_index=True)
        self._write(trades)

    def read(self) -> pd.DataFrame:
        return self._load()
