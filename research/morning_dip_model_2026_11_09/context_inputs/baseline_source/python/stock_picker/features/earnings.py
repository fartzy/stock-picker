"""Earnings-calendar skip for live scoring.

The model is a price-pattern model. A name that reported last night or this
morning is a news day -- skip it rather than pretend recency features explain
the open. One Finnhub calendar request covers the whole universe.
"""

from __future__ import annotations

from datetime import date, timedelta

from stock_picker.ingestion.finnhub_client import fetch_earnings_tickers


def prior_session(as_of: date) -> date:
    """Previous weekday -- AMC reports from Friday still matter on Monday."""
    day = as_of - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def fetch_recent_earnings_tickers(tickers: list[str], as_of: date) -> set[str]:
    """Universe names with earnings on the prior session or `as_of` (BMO today)."""
    return fetch_earnings_tickers(tickers, as_of, from_date=prior_session(as_of))
