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


def feature_tables(tickers: list[str] | None = None) -> dict:
    tickers = tickers if tickers is not None else active_tickers()
    feature_store = FeatureStore()
    return {ticker: feature_store.read(ticker) for ticker in tickers}
