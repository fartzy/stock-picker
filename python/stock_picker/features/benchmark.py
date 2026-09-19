"""SPY day-session returns for a set of dates.

Lets the Trade History UI show "vs S&P" next to each day's own P&L -- a
quick eyeball gut-check against the market, since a good-looking day's
return means less if the whole market was up just as much (or more).
"""

from __future__ import annotations

import pandas as pd

from stock_picker.ingestion.yfinance_client import download_price_history

BENCHMARK_TICKER = "SPY"


def _spy_history():
    return download_price_history([BENCHMARK_TICKER]).get(BENCHMARK_TICKER)


def fetch_benchmark_returns(dates: list[str]) -> dict[str, float]:
    """Day-session (open->close) return for SPY on each requested date
    (ISO "YYYY-MM-DD"). Dates with no matching trading day (weekends,
    holidays, or older than the fetch window) are simply omitted, not
    erred on -- the caller only ever asks for dates a real trade happened,
    so a miss here is a data-availability gap, not a bug to surface."""
    if not dates:
        return {}

    history = _spy_history()
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


def fetch_benchmark_overnight(dates: list[str]) -> dict[str, float]:
    """Close-to-close if you did not sell: prior session close -> this close.

    Includes the overnight gap plus the cash session. Dates with no prior
    close in the fetch window are omitted.
    """
    if not dates:
        return {}
    history = _spy_history()
    if history is None or history.empty or "Close" not in history.columns:
        return {}
    closes = history["Close"].copy()
    closes.index = history.index.strftime("%Y-%m-%d")
    closes = closes[closes.notna() & (closes > 0)]
    if closes.empty:
        return {}
    prior = closes.shift(1)
    overnight = {}
    for requested_date in dates:
        if requested_date not in closes.index or requested_date not in prior.index:
            continue
        prev = prior.loc[requested_date]
        today = closes.loc[requested_date]
        if pd.isna(prev) or float(prev) <= 0:
            continue
        overnight[requested_date] = float(today / prev) - 1.0
    return overnight


def fetch_benchmark_hold(start: str, end: str) -> dict | None:
    """SPY close-to-close if you bought at the first day's close and held
    through the last day's close. Not an average of session returns."""
    if not start or not end:
        return None
    history = _spy_history()
    if history is None or history.empty or "Close" not in history.columns:
        return None
    closes = history["Close"].copy()
    closes.index = history.index.strftime("%Y-%m-%d")
    closes = closes[closes.notna() & (closes > 0)]
    if closes.empty:
        return None
    on_or_after_start = closes[closes.index >= start]
    on_or_before_end = closes[closes.index <= end]
    if on_or_after_start.empty or on_or_before_end.empty:
        return None
    start_close = float(on_or_after_start.iloc[0])
    end_close = float(on_or_before_end.iloc[-1])
    start_day = str(on_or_after_start.index[0])
    end_day = str(on_or_before_end.index[-1])
    return {"from": start_day, "to": end_day, "return": (end_close / start_close) - 1.0}
