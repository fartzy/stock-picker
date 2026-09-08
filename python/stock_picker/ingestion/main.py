"""Entrypoint: pull the top-2000-by-market-cap universe and store 6mo OHLCV history."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.tickers.manual_additions import MANUAL_ADDITIONS
from stock_picker.tickers.nasdaq_directory import fetch_candidate_tickers
from stock_picker.tickers.universe import build_universe

# Conservative starting point against Yahoo's undocumented rate limits on the
# .info endpoint -- tune down if 429-driven None results start showing up.
_DEFAULT_MAX_WORKERS = 20


def _fetch_one_market_cap(ticker: str) -> tuple[str, float | None]:
    try:
        return ticker, yf.Ticker(ticker).info.get("marketCap")
    except Exception:
        # yfinance's failure modes for delisted/renamed/unrecognized tickers
        # aren't a stable contract -- broad catch so one bad ticker (of
        # thousands, at this universe size) can never abort the whole run.
        return ticker, None


def _fetch_market_caps(tickers: list[str], max_workers: int = _DEFAULT_MAX_WORKERS) -> dict[str, float]:
    market_caps: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one_market_cap, ticker) for ticker in tickers]
        for future in as_completed(futures):
            ticker, market_cap = future.result()
            if market_cap:
                market_caps[ticker] = market_cap
    return market_caps


def main() -> None:
    candidates = fetch_candidate_tickers()
    market_caps = _fetch_market_caps(candidates)
    universe = build_universe(market_caps, n=2000, manual_additions=MANUAL_ADDITIONS)

    UniverseStore().sync(universe)

    price_history = download_price_history(list(universe))

    price_store = PriceStore()
    for ticker, history in price_history.items():
        price_store.write(ticker, history)


if __name__ == "__main__":
    main()
