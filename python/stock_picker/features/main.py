"""Entrypoint: compute and persist features for every actively tracked ticker."""

from __future__ import annotations

import pandas as pd

from stock_picker.features.pipeline import build_features_for_universe
from stock_picker.features.regime import VIX_TICKER
from stock_picker.ingestion.session import completed_sessions, last_completed_session_date
from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore

BENCHMARK_TICKER = "SPY"


def main() -> None:
    tickers = UniverseStore().active_tickers()

    price_store = PriceStore()
    cutoff = last_completed_session_date()
    print(f"features through last completed session {cutoff}", flush=True)
    histories = {}
    for ticker in tickers:
        try:
            history = price_store.read(ticker)
        except FileNotFoundError:
            # Active in UniverseStore doesn't guarantee ingestion succeeded for it
            # (e.g. a transient yfinance failure) -- skip rather than crash the
            # whole run over one ticker.
            print(f"skipping {ticker}: no price data found")
            continue
        history = completed_sessions(history)
        if history.empty:
            print(f"skipping {ticker}: no completed session through {cutoff}")
            continue
        histories[ticker] = history

    downloaded = download_price_history([BENCHMARK_TICKER, VIX_TICKER])
    benchmark_history = completed_sessions(downloaded[BENCHMARK_TICKER])
    vix_history = completed_sessions(downloaded.get(VIX_TICKER, pd.DataFrame()))
    universe_store = UniverseStore()
    sector_by_ticker = universe_store.sector_by_ticker()

    features_by_ticker = build_features_for_universe(
        histories,
        benchmark_history=benchmark_history,
        sector_by_ticker=sector_by_ticker or None,
        spy_history=benchmark_history,
        vix_history=vix_history if not vix_history.empty else None,
    )

    feature_store = FeatureStore()
    for ticker, features in features_by_ticker.items():
        feature_store.write(ticker, features)


if __name__ == "__main__":
    main()
