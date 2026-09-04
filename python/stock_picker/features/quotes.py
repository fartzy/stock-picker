"""Live quote wiring and shaping for the API.

Wiring + pure transform split, mirroring trades.py -- quote_summaries() is
unit-testable with a plain dict, no yfinance/network mocking needed.
"""

from __future__ import annotations

from stock_picker.ingestion.yfinance_client import fetch_quotes

DIFF_DECIMAL_PLACES = 2
PCT_DECIMAL_PLACES = 4


def fetch_ticker_quotes(tickers: list[str]) -> dict[str, dict]:
    return fetch_quotes(tickers)


def quote_summaries(raw_quotes: dict[str, dict]) -> list[dict]:
    """Per ticker: open, last, diff = last - open, diff_pct, and (when a
    previous close is available) the overnight gap and its own diff/pct.

    Tickers missing from `raw_quotes` (yfinance couldn't fetch them) are
    simply omitted, matching fetch_quotes' own failure handling.
    """
    summaries = []
    for ticker, quote in raw_quotes.items():
        open_price = quote["open"]
        last_price = quote["last"]
        diff = round(last_price - open_price, DIFF_DECIMAL_PLACES)
        diff_pct = round(diff / open_price, PCT_DECIMAL_PLACES) if open_price else 0.0
        summary = {
            "ticker": ticker,
            "open": open_price,
            "last": last_price,
            "diff": diff,
            "diff_pct": diff_pct,
            "prev_close": None,
            "gap": None,
            "gap_pct": None,
        }
        prev_close = quote.get("prev_close")
        if prev_close:
            gap = round(open_price - prev_close, DIFF_DECIMAL_PLACES)
            summary["prev_close"] = prev_close
            summary["gap"] = gap
            summary["gap_pct"] = round(gap / prev_close, PCT_DECIMAL_PLACES)
        summaries.append(summary)
    return summaries
