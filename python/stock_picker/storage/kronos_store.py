"""Append-only Kronos observations and resumable collection state (no model weights)."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from stock_picker.storage.paths import data_root
from tools.kronos.contract import digest, manifest_digest

DEFAULT_DIRECTORY = data_root() / "kronos"
FEATURE_VERSION = 1
PRICE_BASIS = "price_store_ohlcv_as_stored"
TERMINAL_STATUSES = ("ok", "invalid_forecast", "missing_context")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def contract_from_description(description: dict) -> dict:
    return {
        "execution_fingerprint": description["execution_fingerprint"],
        "manifest_hash": manifest_digest(description["manifest"]),
        "feature_version": FEATURE_VERSION,
        "price_basis": PRICE_BASIS,
    }


class KronosStore:
    """One connection per operation so the store is process-safe and picklable.

    Read paths never create a database. Profiles and raw observations are immutable;
    the small attempt table tracks the current input revision and retryable failures.
    """

    def __init__(self, data_dir: Path | str = DEFAULT_DIRECTORY):
        self.directory = Path(data_dir)
        self.path = self.directory / "forecasts.sqlite3"

    @contextmanager
    def _connect(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                contract_id TEXT PRIMARY KEY, contract TEXT NOT NULL, description TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS forecasts (
                request_key TEXT PRIMARY KEY, contract_id TEXT NOT NULL,
                ticker TEXT NOT NULL, target TEXT NOT NULL, input_hash TEXT NOT NULL,
                payload TEXT NOT NULL, comparison TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS forecast_target ON forecasts(contract_id,ticker,target);
            CREATE TABLE IF NOT EXISTS attempts (
                contract_id TEXT NOT NULL, ticker TEXT NOT NULL, target TEXT NOT NULL,
                status TEXT NOT NULL, request_key TEXT, detail TEXT, observed_at TEXT NOT NULL,
                PRIMARY KEY(contract_id,ticker,target));
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _read(self) -> sqlite3.Connection | None:
        if not self.path.exists():
            return None
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)

    def register_profile(self, description: dict) -> dict:
        profile = description["execution_profile"]
        if digest(profile) != description["execution_fingerprint"]:
            raise ValueError("Kronos profile fingerprint mismatch")
        contract = contract_from_description(description)
        # Startup latency is an observation, not immutable execution identity.
        immutable = {key: description[key] for key in ("execution_profile", "execution_fingerprint", "manifest")}
        with self._connect() as db:
            prior = db.execute("SELECT description FROM profiles WHERE contract_id=?", (digest(contract),)).fetchone()
            if prior and prior[0] != canonical(immutable):
                raise ValueError("conflicting Kronos profile")
            db.execute("INSERT OR IGNORE INTO profiles VALUES (?,?,?)", (digest(contract), canonical(contract), canonical(immutable)))
        return contract

    def description(self, contract: dict) -> dict | None:
        db = self._read()
        if db is None:
            return None
        with db:
            row = db.execute("SELECT description FROM profiles WHERE contract_id=?", (digest(contract),)).fetchone()
        db.close()
        return json.loads(row[0]) if row else None

    def set_collection_contract(self, contract: dict) -> None:
        """Explicit CLI selection; never silently replace an established device/profile."""
        current = self.collection_contract()
        if current is not None and current != contract:
            raise ValueError("collection contract already pinned; use a separate --cache-dir for a different runtime")
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO settings VALUES ('collection_contract',?)", (canonical(contract),))

    def collection_contract(self) -> dict | None:
        db = self._read()
        if db is None:
            return None
        with db:
            row = db.execute("SELECT value FROM settings WHERE key='collection_contract'").fetchone()
        db.close()
        return json.loads(row[0]) if row else None

    def record_attempt(self, contract: dict, ticker: str, target: str, status: str,
                       request_key: str | None = None, detail: str | None = None) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO attempts VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(contract_id,ticker,target) DO UPDATE SET
                status=excluded.status, request_key=excluded.request_key,
                detail=excluded.detail, observed_at=excluded.observed_at""",
                (digest(contract), ticker, target, status, request_key, detail, datetime.now(timezone.utc).isoformat()))

    def save(self, contract: dict, prepared: dict, result: dict, source: dict) -> dict:
        for key in ("ticker", "target_session", "input_hash", "request_key", "context_end", "price_basis"):
            if result.get(key) != prepared[key]:
                raise ValueError(f"Kronos response mismatch: {key}")
        for key in ("execution_fingerprint", "manifest_hash"):
            if result.get(key) != contract[key]:
                raise ValueError(f"Kronos response mismatch: {key}")
        if result.get("status") not in ("ok", "invalid_forecast") or not result.get("generated_at"):
            raise ValueError("Kronos response lacks forecast provenance")
        payload = {**result, "last_context_close": prepared["bars"][-1]["close"],
                   "source": source, "stored_at": datetime.now(timezone.utc).isoformat()}
        # A rerun cannot replace the first observation time. Only computation
        # payloads are compared; timing and source-file container bytes can differ.
        comparison = {key: value for key, value in payload.items() if key not in {
            "generated_at", "reconstructed", "inference_seconds", "stored_at", "source"}}
        with self._connect() as db:
            prior = db.execute("SELECT payload,comparison FROM forecasts WHERE request_key=?", (prepared["request_key"],)).fetchone()
            if prior and prior[1] != canonical(comparison):
                raise ValueError("conflicting same-key Kronos forecast")
            if prior:
                payload = json.loads(prior[0])
            else:
                db.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)", (
                    prepared["request_key"], digest(contract), prepared["ticker"], prepared["target_session"],
                    prepared["input_hash"], canonical(payload), canonical(comparison)))
        self.record_attempt(contract, prepared["ticker"], prepared["target_session"], payload["status"], prepared["request_key"])
        return payload

    def get(self, contract: dict, ticker: str, target: str, request_key: str | None = None) -> dict | None:
        db = self._read()
        if db is None:
            return None
        with db:
            if request_key is None:
                row = db.execute("""SELECT f.payload FROM attempts a JOIN forecasts f ON f.request_key=a.request_key
                    WHERE a.contract_id=? AND a.ticker=? AND a.target=? AND a.status IN ('ok','invalid_forecast')""",
                    (digest(contract), ticker, target)).fetchone()
            else:
                row = db.execute("SELECT payload FROM forecasts WHERE contract_id=? AND ticker=? AND target=? AND request_key=?",
                                 (digest(contract), ticker, target, request_key)).fetchone()
        db.close()
        return json.loads(row[0]) if row else None

    def records_for_ticker(self, contract: dict, ticker: str) -> dict[str, dict]:
        """Load a ticker's exact-key revisions once for historical materialization."""
        db = self._read()
        if db is None:
            return {}
        with db:
            rows = db.execute("SELECT request_key,payload FROM forecasts WHERE contract_id=? AND ticker=?",
                              (digest(contract), ticker)).fetchall()
        db.close()
        return {key: json.loads(payload) for key, payload in rows}

    def coverage(self, contract: dict, tickers: list[str], sessions: list[str]) -> dict:
        counts = {status: 0 for status in TERMINAL_STATUSES}
        populated_sessions = set()
        db = self._read()
        if db is not None and tickers and sessions:
            with db:
                # Temporary tables avoid SQLite's bound-parameter limit for large universes.
                db.execute("CREATE TEMP TABLE wanted_tickers (ticker TEXT PRIMARY KEY)")
                db.executemany("INSERT INTO wanted_tickers VALUES (?)", ((ticker,) for ticker in set(tickers)))
                rows = db.execute("""SELECT target,status,count(*) FROM attempts
                    WHERE contract_id=? AND ticker IN (SELECT ticker FROM wanted_tickers)
                    AND target>=? AND target<=? GROUP BY target,status""",
                    (digest(contract), sessions[0], sessions[-1])).fetchall()
                allowed = set(sessions)
                for target, status, count in rows:
                    if target in allowed:
                        counts[status] = counts.get(status, 0) + count
                        if status == "ok" and count:
                            populated_sessions.add(target)
            db.close()
        expected = len(set(tickers)) * len(sessions)
        completed = sum(counts.get(status, 0) for status in TERMINAL_STATUSES)
        return {"expected": expected, "completed": completed, "statuses": counts,
                "ready": expected > 0 and completed == expected and len(populated_sessions) == len(sessions),
                "populated_sessions": len(populated_sessions),
                "start": sessions[0] if sessions else None, "end": sessions[-1] if sessions else None,
                "tickers": len(set(tickers)), "sessions": len(sessions)}

    def completed_keys(self, contract: dict, start: str, end: str) -> set[tuple[str, str]]:
        """Cheap resume index; missing contexts are retried when sources improve."""
        db = self._read()
        if db is None:
            return set()
        with db:
            rows = db.execute("""SELECT ticker,target FROM attempts WHERE contract_id=?
                AND target>=? AND target<=? AND status IN ('ok','invalid_forecast')""",
                (digest(contract), start, end)).fetchall()
        db.close()
        return set(rows)

    def first_target(self, contract: dict) -> str | None:
        db = self._read()
        if db is None:
            return None
        with db:
            row = db.execute("SELECT min(target) FROM attempts WHERE contract_id=?", (digest(contract),)).fetchone()
        db.close()
        return row[0] if row else None
