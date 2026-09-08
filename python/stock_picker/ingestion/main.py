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

# Empirically found too high: 20 workers hitting the full .info endpoint
# triggered widespread "Invalid Crumb" 401s and then outright 429 rate-limits
# from Yahoo's undocumented API, corrupting most of the fetch (silently
# returning None for the majority of candidates). 5 is a much more
# conservative starting point -- tune down further if 429s still show up.
_DEFAULT_MAX_WORKERS = 5


def _fetch_one_market_cap(ticker: str) -> tuple[str, float | None]:
    try:
        # fast_info is a lighter-weight endpoint than .info (which pulls
        # financials/holders/recommendations/etc. this doesn't need) --
        # fewer, smaller requests per ticker means less exposure to the
        # rate-limit/crumb-invalidation behavior above.
        return ticker, yf.Ticker(ticker).fast_info.market_cap
    except Exception:
        # yfinance's failure modes for delisted/renamed/unrecognized tickers,
        # and now also rate-limiting, aren't a stable contract -- broad catch
        # so one bad ticker (of thousands, at this universe size) can never
        # abort the whole run.
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
