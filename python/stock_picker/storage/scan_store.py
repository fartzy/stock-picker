"""Repository for morning scan payloads (fit list and rank list).

Same DI as TradeStore: `ScanStore(data_dir=tmp_path)`. Persistence is a
swappable backend. SQLite is the source of truth (one row per day+kind);
dated JSON under the same directory stays the human dump Trading already
knows. Feature parquet is not involved.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "buy_signals"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    as_of TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (as_of, kind)
);
"""


class ScanLogBackend(Protocol):
    def read(self, as_of: str, kind: str) -> dict | None: ...

    def write(self, as_of: str, kind: str, payload: dict) -> None: ...

    def days(self) -> list[str]: ...


class JsonScanLog:
    """Dated JSON files. `as_of` for rank is the day; kind selects the name."""

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, as_of: str, kind: str) -> Path:
        name = f"{as_of}-rank.json" if kind == "rank" else f"{as_of}.json"
        return self._data_dir / name

    def read(self, as_of: str, kind: str) -> dict | None:
        path = self._path(as_of, kind)
        if not path.is_file():
            latest = self._data_dir / "latest.json"
            path = latest if kind == "fit" and latest.is_file() else path
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("as_of") != as_of:
            return None
        return payload

    def write(self, as_of: str, kind: str, payload: dict) -> None:
        path = self._path(as_of, kind)
        path.write_text(json.dumps(payload, indent=2))
        if kind == "fit":
            (self._data_dir / "latest.json").write_text(json.dumps(payload, indent=2))

    def days(self) -> list[str]:
        found = set()
        for path in self._data_dir.glob("*.json"):
            if path.name == "latest.json":
                continue
            stem = path.name.removesuffix(".json").removesuffix("-rank")
            if len(stem) == 10 and stem[0].isdigit():
                found.add(stem)
        return sorted(found)


class SqliteScanLog:
    """One row per (as_of, kind). Also writes the JSON dump so git/picks stay."""

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "scans.db"
        self._json = JsonScanLog(self._data_dir)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        self._import_json_if_empty()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _import_json_if_empty(self) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            if count:
                return
        for path in sorted(self._data_dir.glob("*.json")):
            if path.name == "latest.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "as_of" not in payload:
                continue
            kind = payload.get("kind") or ("rank" if path.name.endswith("-rank.json") else "fit")
            as_of = payload["as_of"]
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO scans (as_of, kind, payload) VALUES (?, ?, ?)",
                    (as_of, kind, json.dumps(payload)),
                )

    def read(self, as_of: str, kind: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM scans WHERE as_of = ? AND kind = ?",
                (as_of, kind),
            ).fetchone()
        if row:
            payload = json.loads(row["payload"])
            if isinstance(payload, dict) and payload.get("as_of") == as_of:
                return payload
        return self._json.read(as_of, kind)

    def write(self, as_of: str, kind: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scans (as_of, kind, payload) VALUES (?, ?, ?)",
                (as_of, kind, json.dumps(payload)),
            )
        self._json.write(as_of, kind, payload)

    def days(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT as_of FROM scans ORDER BY as_of").fetchall()
        sqlite_days = [row["as_of"] for row in rows]
        return sorted(set(sqlite_days) | set(self._json.days()))


def default_scan_backend(data_dir: Path | str) -> ScanLogBackend:
    return SqliteScanLog(data_dir)


class ScanStore:
    """Reads and writes one morning scan. Backend defaults to SQLite."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        backend: ScanLogBackend | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend if backend is not None else default_scan_backend(self._data_dir)

    def read(self, as_of: str, kind: str = "fit") -> dict | None:
        return self._backend.read(as_of, kind)

    def write(self, as_of: str, kind: str, payload: dict) -> None:
        self._backend.write(as_of, kind, payload)

    def days(self) -> list[str]:
        return self._backend.days()
