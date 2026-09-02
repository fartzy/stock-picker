"""Thin wrapper around yfinance for batch OHLCV history pulls."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_price_history(
    tickers: list[str],
    period: str = "6mo",
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
    # special case.
    return {ticker: raw[ticker].dropna(how="all") for ticker in tickers if ticker in raw}
