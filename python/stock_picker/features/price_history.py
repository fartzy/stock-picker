"""Price history wiring and shaping for the API -- daily (from the already-
ingested PriceStore) and intraday (fetched live from yfinance, never
persisted: hourly retention is ~730 days, cheap to refetch, unlike the daily
history features are trained on).
"""

from __future__ import annotations

import pandas as pd

from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.storage.price_store import PriceStore

INTRADAY_PERIOD = "5d"
INTRADAY_INTERVAL = "1h"


def daily_price_history(ticker: str) -> pd.DataFrame:
    return PriceStore().read(ticker)


def intraday_price_history(ticker: str) -> pd.DataFrame:
    history = download_price_history([ticker], period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
    if ticker not in history:
        raise ValueError(f"no intraday data for {ticker}")
    return history[ticker]


def price_series(history: pd.DataFrame) -> list[dict]:
    """Ascending-by-date OHLCV rows, JSON-serializable."""
    sorted_history = history.sort_index()
    return [
        {
            "date": index.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for index, row in sorted_history.iterrows()
    ]
