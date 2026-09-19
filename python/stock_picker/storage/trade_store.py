"""Repository for logged trade executions.

Callers only see `TradeStore.append` / `read` -- the same DI as every other
store (`TradeStore(data_dir=tmp_path)`). Persistence is a swappable backend
(strategy): SQLite for the live log (row identity, unique fills), parquet+csv
as the fallback / human dump. Other transactional stores (scans, training
runs) can reuse the same pattern without touching API or P&L code.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "trades"

_COLUMNS = ["ticker", "side", "shares", "price", "executed_at"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    executed_at TEXT NOT NULL,
    UNIQUE (ticker, side, shares, price, executed_at)
);
"""


@dataclass
class Trade:
    ticker: str
    side: str  # "buy" | "sell"
    shares: float
    price: float
    executed_at: str  # ISO 8601 with UTC offset, e.g. datetime.now().astimezone().isoformat()


class TradeLogBackend(Protocol):
    """Strategy: how the log is stored. TradeStore is the repository."""

    def read(self) -> pd.DataFrame: ...

    def append(self, trade: Trade) -> None: ...


class ParquetTradeLog:
    """Rewrite-the-file parquet + sidecar csv. Fine for a handful of rows;
    no unique constraint -- duplicates are the caller's problem."""

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "trades.parquet"
        self._csv_path = self._data_dir / "trades.csv"

    def read(self) -> pd.DataFrame:
        if self._path.exists():
            return pd.read_parquet(self._path)
        return pd.DataFrame(columns=_COLUMNS)

    def append(self, trade: Trade) -> None:
        trades = self.read()
        new_row = pd.DataFrame([asdict(trade)])
        trades = new_row if trades.empty else pd.concat([trades, new_row], ignore_index=True)
        self._write(trades)

    def _write(self, trades: pd.DataFrame) -> None:
        trades.to_parquet(self._path, index=False)
        trades.to_csv(self._csv_path, index=False)


class SqliteTradeLog:
    """One row per fill. Duplicate (ticker, side, shares, price, executed_at)
    is a no-op. Sidecar csv stays a human dump, not the source of truth.
    Imports an existing trades.parquet once if the table is empty.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "stockpicker.db"
        self._csv_path = self._data_dir / "trades.csv"
        self._parquet_path = self._data_dir / "trades.parquet"
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
        self._import_parquet_if_empty()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _import_parquet_if_empty(self) -> None:
        if not self._parquet_path.exists():
            return
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if count:
                return
        frame = pd.read_parquet(self._parquet_path)
        if frame.empty:
            return
        rows = [
            (row["ticker"], row["side"], float(row["shares"]), float(row["price"]), row["executed_at"])
            for row in frame[_COLUMNS].to_dict(orient="records")
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO trades (ticker, side, shares, price, executed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        self._write_csv(self.read())

    def read(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, side, shares, price, executed_at FROM trades ORDER BY id"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame([dict(row) for row in rows], columns=_COLUMNS)

    def append(self, trade: Trade) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trades (ticker, side, shares, price, executed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (trade.ticker, trade.side, trade.shares, trade.price, trade.executed_at),
            )
        self._write_csv(self.read())

    def _write_csv(self, trades: pd.DataFrame) -> None:
        trades.to_csv(self._csv_path, index=False)


def default_trade_backend(data_dir: Path | str) -> TradeLogBackend:
    return SqliteTradeLog(data_dir)


class TradeStore:
    """Reads and appends the trade log. Backend defaults to SQLite."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        backend: TradeLogBackend | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend if backend is not None else default_trade_backend(self._data_dir)

    def append(self, trade: Trade) -> None:
        self._backend.append(trade)

    def read(self) -> pd.DataFrame:
        return self._backend.read()
