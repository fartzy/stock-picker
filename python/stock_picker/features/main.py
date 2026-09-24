"""Entrypoint: compute and persist features for every actively tracked ticker."""

from __future__ import annotations

import pandas as pd

from stock_picker.features.news import articles_by_ticker
from stock_picker.features.pipeline import build_features_for_universe
from stock_picker.features.regime import VIX_TICKER
from stock_picker.log import get_logger
from stock_picker.ingestion.session import completed_sessions, last_completed_session_date
from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore

BENCHMARK_TICKER = "SPY"

logger = get_logger(__name__)


def main() -> None:
    tickers = UniverseStore().active_tickers()

    price_store = PriceStore()
    cutoff = last_completed_session_date()
    logger.info("features through last completed session %s", cutoff)
    histories = {}
    missing_prices: list[str] = []
    for ticker in tickers:
        try:
            history = price_store.read(ticker)
        except FileNotFoundError:
            # Active in UniverseStore doesn't guarantee ingestion succeeded for it
            # (e.g. a transient yfinance failure) -- skip rather than crash the
            # whole run over one ticker. Nightly already retried these on the
            # daily pull and Finnhub candle backfill; leftover names still have no bar.
            logger.warning("skipping %s: no price data found", ticker)
            missing_prices.append(ticker)
            continue
        history = completed_sessions(history)
        if history.empty:
            logger.warning("skipping %s: no completed session through %s", ticker, cutoff)
            missing_prices.append(ticker)
            continue
        histories[ticker] = history
    if missing_prices:
        logger.warning(
            "features missing completed prices for %s tickers: %s",
            len(missing_prices),
            ", ".join(missing_prices[:20]),
        )

    downloaded = download_price_history([BENCHMARK_TICKER, VIX_TICKER])
    spy = downloaded.get(BENCHMARK_TICKER)
    if spy is None or spy.empty:
        try:
            spy = price_store.read(BENCHMARK_TICKER)
            logger.warning("SPY download missed -- using stored %s", BENCHMARK_TICKER)
        except FileNotFoundError:
            spy = pd.DataFrame()
            logger.warning("SPY missing -- regime columns empty")
    benchmark_history = completed_sessions(spy) if not spy.empty else pd.DataFrame()
    vix_raw = downloaded.get(VIX_TICKER, pd.DataFrame())
    vix_history = completed_sessions(vix_raw) if not vix_raw.empty else pd.DataFrame()
    universe_store = UniverseStore()
    sector_by_ticker = universe_store.sector_by_ticker()

    from stock_picker.ingestion.weather import read_nyc_weather, refresh_nyc_weather

    try:
        refresh_nyc_weather()
    except Exception:
        logger.exception("NYC weather refresh failed -- using stored series if any")
    nyc_weather = read_nyc_weather()

    news_by_ticker = articles_by_ticker()
    features_by_ticker = build_features_for_universe(
        histories,
        benchmark_history=benchmark_history,
        sector_by_ticker=sector_by_ticker or None,
        spy_history=benchmark_history,
        vix_history=vix_history if not vix_history.empty else None,
        news_articles_by_ticker=news_by_ticker,
        nyc_weather=nyc_weather if not nyc_weather.empty else None,
    )

    feature_store = FeatureStore()
    for ticker, features in features_by_ticker.items():
        feature_store.write(ticker, features)


if __name__ == "__main__":
    main()
