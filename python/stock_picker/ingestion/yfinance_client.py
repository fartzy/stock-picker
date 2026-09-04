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


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """Today's open + latest price, plus the prior session's close, per ticker.

    period="2d" so there's a previous-close row to compute the overnight gap
    from, in addition to today's open/latest -- period="1d" only ever
    returns today's single row.
    """
    history = download_price_history(tickers, period="2d", interval="1d")
    quotes = {}
    for ticker, df in history.items():
        quote = {"open": float(df["Open"].iloc[-1]), "last": float(df["Close"].iloc[-1])}
        if len(df) > 1:
            quote["prev_close"] = float(df["Close"].iloc[-2])
        quotes[ticker] = quote
    return quotes
