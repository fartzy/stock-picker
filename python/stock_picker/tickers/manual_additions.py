"""Manually curated tickers to track alongside the market-cap-derived universe.

Anything listed here is tracked regardless of market cap ranking -- add
tickers of interest even if they'd never make the top-N cut on their own.
Tracking is permanent once synced (see storage.universe_store.UniverseStore):
removing a ticker from this list just stops it from being re-confirmed on
future syncs, it does not deactivate or delete it.
"""

from __future__ import annotations

MANUAL_ADDITIONS: list[str] = [
    "NVDA",
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "WMT",
    "XOM",
    "UNH",
]
