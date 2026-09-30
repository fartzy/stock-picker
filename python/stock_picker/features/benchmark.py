"""SPY day-session returns for a set of dates.

Lets the Trade History UI show "vs S&P" next to each day's own P&L -- a
quick eyeball gut-check against the market, since a good-looking day's
return means less if the whole market was up just as much (or more).
"""

from __future__ import annotations

import pandas as pd

from stock_picker.ingestion.yfinance_client import download_price_history

BENCHMARK_TICKER = "SPY"


def fetch_benchmark_history() -> pd.DataFrame | None:
    """Fetch one daily SPY history for all comparisons in an API response."""
    return download_price_history([BENCHMARK_TICKER]).get(BENCHMARK_TICKER)


def fetch_benchmark_returns(
    dates: list[str], *, history: pd.DataFrame | None = None
) -> dict[str, float]:
    """Day-session (open->close) return for SPY on each requested date
    (ISO "YYYY-MM-DD"). Dates with no matching trading day (weekends,
    holidays, or older than the fetch window) are simply omitted, not
    erred on -- the caller only ever asks for dates a real trade happened,
    so a miss here is a data-availability gap, not a bug to surface."""
    if not dates:
        return {}

    if history is None:
        history = fetch_benchmark_history()
    if history is None or history.empty or not {"Open", "Close"}.issubset(history.columns):
        return {}

    day_strings = history.index.strftime("%Y-%m-%d")
    returns = {}
    for requested_date in dates:
        matches = history[day_strings == requested_date]
        if matches.empty:
            continue
        row = matches.iloc[0]
        open_price, close_price = row["Open"], row["Close"]
        if pd.isna(open_price) or pd.isna(close_price) or open_price <= 0 or close_price <= 0:
            continue
        returns[requested_date] = float((close_price - open_price) / open_price)
    return returns


def fetch_benchmark_overnight(
    dates: list[str], *, history: pd.DataFrame | None = None
) -> dict[str, float]:
    """Close-to-close if you did not sell: prior session close -> this close.

    Includes the overnight gap plus the cash session. Dates with no prior
    close in the fetch window are omitted.
    """
    if not dates:
        return {}
    if history is None:
        history = fetch_benchmark_history()
    if history is None or history.empty or "Close" not in history.columns:
        return {}
    sorted_history = history.sort_index()
    closes = sorted_history["Close"].copy()
    closes.index = sorted_history.index.strftime("%Y-%m-%d")
    if closes.empty:
        return {}
    prior = closes.shift(1)
    overnight = {}
    for requested_date in dates:
        if requested_date not in closes.index or requested_date not in prior.index:
            continue
        prev = prior.loc[requested_date]
        today = closes.loc[requested_date]
        if pd.isna(prev) or pd.isna(today) or float(prev) <= 0 or float(today) <= 0:
            continue
        overnight[requested_date] = float(today / prev) - 1.0
    return overnight


def fetch_benchmark_hold(
    start: str, end: str, *, history: pd.DataFrame | None = None
) -> dict | None:
    """SPY close-to-close if you bought at the first day's close and held
    through the last day's close. Not an average of session returns."""
    if not start or not end:
        return None
    if history is None:
        history = fetch_benchmark_history()
    if history is None or history.empty or "Close" not in history.columns:
        return None
    sorted_history = history.sort_index()
    closes = sorted_history["Close"].copy()
    closes.index = sorted_history.index.strftime("%Y-%m-%d")
    if start not in closes.index or end not in closes.index:
        return None
    start_close, end_close = closes.loc[start], closes.loc[end]
    if pd.isna(start_close) or pd.isna(end_close) or start_close <= 0 or end_close <= 0:
        return None
    return {"from": start, "to": end, "return": float(end_close / start_close) - 1.0}
