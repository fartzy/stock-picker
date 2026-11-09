"""Append-only Jev requests and answers, independent of replaceable morning scans."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from stock_picker.storage.paths import data_root

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    session TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS forecasts_session ON forecasts(session, ticker);
CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts
BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts
BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;
"""


class JevForecastStore:
    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else data_root() / "jev"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "forecasts.db"
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._db_path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def append(self, record: dict) -> dict:
        """Identical retries are idempotent; an existing identity can never change."""
        for field in ("id", "ticker", "session", "requested_at"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"forecast requires {field}")
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?, ?)",
                (record["id"], record["ticker"], record["session"], record["requested_at"], payload),
            )
            existing = connection.execute("SELECT payload FROM forecasts WHERE id = ?", (record["id"],)).fetchone()
            if existing[0] != payload:
                raise ValueError("forecast identity already contains different immutable evidence")
        return json.loads(payload)

    def read(self, forecast_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM forecasts WHERE id = ?", (forecast_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_records(self, *, ticker: str | None = None, session: str | None = None) -> list[dict]:
        clauses, values = [], []
        for field, value in (("ticker", ticker), ("session", session)):
            if value is not None:
                clauses.append(f"{field} = ?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM forecasts" + where + " ORDER BY requested_at, id", values,
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
