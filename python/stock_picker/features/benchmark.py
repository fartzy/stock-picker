"""SPY day-session returns for a set of dates.

Lets the Trade History UI show "vs S&P" next to each day's own P&L -- a
quick eyeball gut-check against the market, since a good-looking day's
return means less if the whole market was up just as much (or more).
"""

from __future__ import annotations

from stock_picker.ingestion.yfinance_client import download_price_history

BENCHMARK_TICKER = "SPY"


def fetch_benchmark_returns(dates: list[str]) -> dict[str, float]:
    """Day-session (open->close) return for SPY on each requested date
    (ISO "YYYY-MM-DD"). Dates with no matching trading day (weekends,
    holidays, or older than the fetch window) are simply omitted, not
    erred on -- the caller only ever asks for dates a real trade happened,
    so a miss here is a data-availability gap, not a bug to surface."""
    if not dates:
        return {}

    history = download_price_history([BENCHMARK_TICKER]).get(BENCHMARK_TICKER)
    if history is None:
        return {}

    day_strings = history.index.strftime("%Y-%m-%d")
    returns = {}
    for requested_date in dates:
        matches = history[day_strings == requested_date]
        if matches.empty:
            continue
        row = matches.iloc[0]
        returns[requested_date] = float((row["Close"] - row["Open"]) / row["Open"])
    return returns
