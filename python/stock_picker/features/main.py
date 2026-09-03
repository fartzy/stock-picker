"""Entrypoint: compute and persist features for every actively tracked ticker."""

from __future__ import annotations

from stock_picker.features.pipeline import build_features_for_universe
from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore

BENCHMARK_TICKER = "SPY"


def main() -> None:
    tickers = UniverseStore().active_tickers()

    price_store = PriceStore()
    histories = {ticker: price_store.read(ticker) for ticker in tickers}

    benchmark_history = download_price_history([BENCHMARK_TICKER])[BENCHMARK_TICKER]

    features_by_ticker = build_features_for_universe(
        histories, benchmark_history=benchmark_history
    )

    feature_store = FeatureStore()
    for ticker, features in features_by_ticker.items():
        feature_store.write(ticker, features)


if __name__ == "__main__":
    main()
