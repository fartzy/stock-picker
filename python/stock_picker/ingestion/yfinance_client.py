"""Thin wrapper around yfinance for batch OHLCV history pulls."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_price_history(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV history for a batch of tickers.

    Returns a mapping of ticker -> DataFrame with columns
    [Open, High, Low, Close, Adj Close, Volume], indexed by date.
    """
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    # yfinance always returns MultiIndex (ticker, field) columns when group_by="ticker"
    # is set, even for a single ticker -- always slice per ticker, no single-ticker
    # special case. A ticker yfinance couldn't fetch (delisted, transient API failure,
    # etc.) comes back as an all-NaN slice -- exclude it rather than storing an empty
    # DataFrame that would break downstream assumptions of at least one row.
    result = {}
    for ticker in tickers:
        if ticker not in raw:
            continue
        history = raw[ticker].dropna(how="all")
        if not history.empty:
            result[ticker] = history
    return result
