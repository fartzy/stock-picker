"""Tickers omitted from morning Rank/Fit (and Test run).

Same JSON add/remove shape as PrunedFeatureStore. Universe/prices stay;
scoring skips these names until they are unblocked.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "ticker_blacklist"
DEFAULT_REASON = "blacklisted"


class TickerBlacklistStore:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "blacklist.json"

    def _load(self) -> list[dict]:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return []

    def _save(self, entries: list[dict]) -> None:
        self._path.write_text(json.dumps(entries, indent=2) + "\n")

    def read(self) -> set[str]:
        return {entry["ticker"] for entry in self._load()}

    def read_all(self) -> list[dict]:
        return sorted(self._load(), key=lambda entry: entry["blocked_at"], reverse=True)

    def add(self, ticker: str, reason: str = DEFAULT_REASON) -> None:
        ticker = ticker.strip().upper()
        entries = self._load()
        if any(entry["ticker"] == ticker for entry in entries):
            return
        entries.append(
            {
                "ticker": ticker,
                "reason": reason,
                "blocked_at": datetime.now().astimezone().isoformat(),
            }
        )
        self._save(entries)

    def remove(self, ticker: str) -> None:
        ticker = ticker.strip().upper()
        self._save([entry for entry in self._load() if entry["ticker"] != ticker])


def blacklisted_tickers(data_dir: Path | str | None = None) -> set[str]:
    store = TickerBlacklistStore() if data_dir is None else TickerBlacklistStore(data_dir=data_dir)
    return store.read()
