"""Repository for the paper book: morning-list names held Open->Close.

Same DI as TradeStore / ScanStore. SQLite only -- this is not a feature
column and not a Fidelity fill. One row per (day, list kind, rank).
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "paper_book"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_picks (
    as_of TEXT NOT NULL,
    kind TEXT NOT NULL,
    rank INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    predicted REAL,
    open_price REAL,
    close_price REAL,
    session_return REAL,
    news_flag TEXT,
    news_checked INTEGER NOT NULL DEFAULT 0,
    prev_close REAL,
    PRIMARY KEY (as_of, kind, rank)
);
"""


@dataclass
class PaperPick:
    as_of: str
    kind: str
    rank: int
    ticker: str
    predicted: float | None
    open_price: float | None
    close_price: float | None
    session_return: float | None
    news_flag: str | None = None
    news_checked: int = 0
    prev_close: float | None = None


class PaperBookStore:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "paper_book.db"
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_picks)")}
        if "news_flag" not in columns:
            conn.execute("ALTER TABLE paper_picks ADD COLUMN news_flag TEXT")
        if "news_checked" not in columns:
            conn.execute("ALTER TABLE paper_picks ADD COLUMN news_checked INTEGER NOT NULL DEFAULT 0")
        if "prev_close" not in columns:
            conn.execute("ALTER TABLE paper_picks ADD COLUMN prev_close REAL")

    def replace_all(self, picks: list[PaperPick]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM paper_picks")
            conn.executemany(
                "INSERT INTO paper_picks "
                "(as_of, kind, rank, ticker, predicted, open_price, close_price, session_return, news_flag, news_checked, prev_close) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        p.as_of,
                        p.kind,
                        p.rank,
                        p.ticker,
                        p.predicted,
                        p.open_price,
                        p.close_price,
                        p.session_return,
                        p.news_flag,
                        p.news_checked,
                        p.prev_close,
                    )
                    for p in picks
                ],
            )

    def read(self) -> list[PaperPick]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT as_of, kind, rank, ticker, predicted, open_price, close_price, "
                "session_return, news_flag, news_checked, prev_close "
                "FROM paper_picks ORDER BY as_of, kind, rank"
            ).fetchall()
        return [PaperPick(**dict(row)) for row in rows]
