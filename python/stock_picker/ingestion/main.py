"""Entrypoint: pull the top-500-by-market-cap universe and store 6mo OHLCV history."""

from __future__ import annotations

import yfinance as yf

from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.tickers.manual_additions import MANUAL_ADDITIONS
from stock_picker.tickers.universe import build_universe, fetch_sp500_constituents


def _fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    market_caps = {}
    for ticker in tickers:
        market_cap = yf.Ticker(ticker).info.get("marketCap")
        if market_cap:
            market_caps[ticker] = market_cap
    return market_caps


def main() -> None:
    constituents = fetch_sp500_constituents()
    market_caps = _fetch_market_caps(constituents)
    universe = build_universe(market_caps, n=500, manual_additions=MANUAL_ADDITIONS)

    UniverseStore().sync(universe)

    price_history = download_price_history(list(universe))

    price_store = PriceStore()
    for ticker, history in price_history.items():
        price_store.write(ticker, history)


if __name__ == "__main__":
    main()
