"""Storage-wiring helpers shared by the API routes and the catalog CLI."""

from __future__ import annotations

from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore


def active_tickers() -> list[str]:
    return UniverseStore().active_tickers()


def sample_history(tickers: list[str] | None = None):
    tickers = tickers if tickers is not None else active_tickers()
    if not tickers:
        raise ValueError("No active tickers found; cannot sample history")
    return PriceStore().read(tickers[0])


# Coverage/correlation don't need every ticker -- a slice of the universe is
# enough for "% non-null" and pairwise corr, and reading all ~2000 parquets
# is what made the Feature Store tab hang on "Loading registry...".
STATS_SAMPLE_SIZE = 80


def feature_tables(tickers: list[str] | None = None, limit: int | None = None) -> dict:
    tickers = tickers if tickers is not None else active_tickers()
    if limit is not None:
        tickers = tickers[:limit]
    feature_store = FeatureStore()
    tables = {}
    for ticker in tickers:
        try:
            tables[ticker] = feature_store.read(ticker)
        except FileNotFoundError:
            continue
    return tables
