"""Repository for tracking universe membership over time.

Membership is monotonic: once a ticker has ever qualified for the tracked
universe (top-N by market cap, or manually added), it's synced in as active
and stays that way -- being absent from a later sync (e.g. falling out of
the top-N ranking) does not remove or deactivate it. The 500 cutoff is
just today's entry criterion, not a cap on what we keep tracking. `active`
is retained as a field for possible future explicit/manual removal (e.g. a
delisting), which isn't implemented yet.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "universe"

_COLUMNS = ["source", "first_seen", "last_seen", "active"]


class UniverseStore:
    """Reads and writes the universe membership registry as a Parquet file."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "registry.parquet"

    def _load(self) -> pd.DataFrame:
        if self._path.exists():
            return pd.read_parquet(self._path)
        return pd.DataFrame(columns=_COLUMNS).rename_axis("ticker")

    def sync(self, current_tickers: dict[str, str], as_of: date | None = None) -> pd.DataFrame:
        """Add/refresh today's computed universe into the registry.

        `current_tickers` maps ticker -> source ("market_cap" or "manual").
        New tickers are added as active with today as their first/last seen
        date; already-tracked tickers get `last_seen` bumped to today.
        Tickers absent from `current_tickers` are left untouched -- they
        are never auto-deactivated. Returns the updated registry.
        """
        today = (as_of or date.today()).isoformat()
        registry = self._load()

        for ticker, source in current_tickers.items():
            if ticker in registry.index:
                registry.loc[ticker, ["last_seen", "active"]] = [today, True]
            else:
                registry.loc[ticker] = [source, today, today, True]

        registry.to_parquet(self._path, index=True)
        return registry.reset_index(names="ticker")

    def active_tickers(self) -> list[str]:
        registry = self._load()
        if registry.empty:
            return []
        return registry.index[registry["active"]].tolist()

    def all_tickers(self) -> pd.DataFrame:
        return self._load().reset_index(names="ticker")
