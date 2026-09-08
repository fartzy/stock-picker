"""Entrypoint: pull the top-2000-by-market-cap universe and store 6mo OHLCV history."""

from __future__ import annotations

import yfinance as yf

from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.tickers.manual_additions import MANUAL_ADDITIONS
from stock_picker.tickers.nasdaq_directory import fetch_candidate_tickers
from stock_picker.tickers.universe import build_universe

# Deliberately NOT threaded, even though this fetches thousands of tickers.
# yfinance 1.7.0's cookie/crumb handling is a SingletonMeta guarded by one
# shared threading.Lock() -- every quoteSummary request (what fast_info hits
# under the hood) funnels through it. Concurrent workers don't parallelize
# here, they queue on that lock; and once Yahoo 429s the crumb fetch once,
# every later request 404s on the stale crumb and re-mints it again under
# that same lock with a 30s timeout ceiling -- N threads doing that back-to-
# back is *slower* than one, not faster, and was observed hanging silently
# for ~10 minutes before the process was killed. A plain sequential loop
# hits the exact same rate limit but without the lock-convoy amplification.


def _fetch_one_market_cap(ticker: str) -> tuple[str, float | None]:
    try:
        # fast_info is a lighter-weight endpoint than .info (which pulls
        # financials/holders/recommendations/etc. this doesn't need).
        return ticker, yf.Ticker(ticker).fast_info.market_cap
    except Exception:
        # yfinance's failure modes for delisted/renamed/unrecognized tickers,
        # and rate-limiting, aren't a stable contract -- broad catch so one
        # bad ticker (of thousands, at this universe size) can never abort
        # the whole run.
        return ticker, None


def _fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    market_caps: dict[str, float] = {}
    for ticker in tickers:
        _, market_cap = _fetch_one_market_cap(ticker)
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
