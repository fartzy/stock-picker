"""Ticker universe: top-500-by-market-cap constituents.

There is no single free data source that cleanly exposes "all US tickers
ranked by market cap." As a starting universe we use the S&P 500
constituents (already the ~500 largest US companies) scraped from
Wikipedia, then re-rank by live market cap pulled via `yfinance` and trim
to the top N.
"""

from __future__ import annotations

import pandas as pd

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_constituents(source_url: str = WIKIPEDIA_SP500_URL) -> list[str]:
    """Return the current S&P 500 ticker symbols, used as the starting universe."""
    tables = pd.read_html(source_url)
    constituents = tables[0]
    return constituents["Symbol"].str.replace(".", "-", regex=False).tolist()


def top_n_by_market_cap(market_caps: dict[str, float], n: int = 500) -> list[str]:
    """Rank tickers by market cap (descending) and return the top `n` symbols."""
    ranked = sorted(market_caps.items(), key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in ranked[:n]]


def build_universe(
    market_caps: dict[str, float],
    n: int = 500,
    manual_additions: list[str] | None = None,
) -> dict[str, str]:
    """Combine the top-N-by-market-cap ranking with manually curated tickers.

    Returns a mapping of ticker -> source ("market_cap" or "manual"),
    suitable for `storage.universe_store.UniverseStore.sync`.
    """
    universe = {ticker: "market_cap" for ticker in top_n_by_market_cap(market_caps, n=n)}
    for ticker in manual_additions or []:
        universe.setdefault(ticker, "manual")
    return universe
