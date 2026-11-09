"""Append-only observations of Jev forecast outcomes; forecasts never change."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from stock_picker.storage.paths import data_root


class JevOutcomeStore:
    def __init__(self, data_dir: Path | str | None = None) -> None:
        root = Path(data_dir) if data_dir is not None else data_root() / "jev"
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / "outcomes.db"
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS outcomes (
                id TEXT PRIMARY KEY, forecast_id TEXT NOT NULL,
                observed_at TEXT NOT NULL, payload TEXT NOT NULL
            )""")
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS outcomes_no_update BEFORE UPDATE ON outcomes
                BEGIN SELECT RAISE(ABORT, 'outcomes are append only'); END;
                CREATE TRIGGER IF NOT EXISTS outcomes_no_delete BEFORE DELETE ON outcomes
                BEGIN SELECT RAISE(ABORT, 'outcomes are append only'); END;
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def append(self, outcome: dict) -> None:
        payload = json.dumps(outcome, sort_keys=True, allow_nan=False)
        identity = hashlib.sha256(payload.encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO outcomes VALUES (?, ?, ?, ?)",
                         (identity, outcome["forecast_id"], outcome["observed_at"], payload))

    def latest(self) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT forecast_id, payload FROM outcomes ORDER BY observed_at, rowid").fetchall()
        return {identity: json.loads(payload) for identity, payload in rows}
