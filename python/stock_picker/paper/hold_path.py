"""Later-day bars for a paper pick, starting the session after its close.

The paper book stores that day's open and close. A hold-after-close rule
needs the next sessions' high and low, which stay in the price files.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_picker.storage.price_store import PriceStore


def later_sessions(
    ticker: str,
    as_of: date,
    days: int,
    prices: PriceStore | None = None,
) -> list[dict]:
    """Up to `days` completed sessions after `as_of`. Missing files return none."""
    if days < 1:
        return []
    store = prices or PriceStore()
    try:
        history = store.read(ticker)
    except FileNotFoundError:
        return []
    if history.empty or "High" not in history.columns or "Low" not in history.columns:
        return []
    stamps = pd.to_datetime(history.index).normalize()
    cutoff = pd.Timestamp(as_of)
    later = history.loc[stamps > cutoff].head(days)
    rows = []
    for stamp, row in later.iterrows():
        high = row.get("High")
        low = row.get("Low")
        close = row.get("Close")
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            continue
        rows.append({
            "date": pd.Timestamp(stamp).date().isoformat(),
            "high": float(high),
            "low": float(low),
            "close": float(close),
        })
    return rows
