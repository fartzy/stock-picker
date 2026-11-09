"""Counterfactual session: buy 8:40 AM CT, sell 2:55 PM CT.

Not "your fill vs the close". 8:40 CT is 9:40 ET (ten minutes after the
bell); 2:55 CT is 15:55 ET (five minutes before the close). Daily Open and
Close are the stored prints for that window -- we do not refetch 1-minute
bars. Only shares bought before 9:00 AM CT qualify. In-progress today stays blank until
the cash session ends; today's aggregate close fills until nightly writes
the settled bar.
"""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.storage.price_store import PriceStore

_NY = ZoneInfo("America/New_York")
_CHICAGO = ZoneInfo("America/Chicago")
HOLD_ENTRY_CUTOFF = time(9, 0)
NOTIONAL_DECIMAL_PLACES = 2


def _buy_session_day(buy_time: str | None) -> str | None:
    if not buy_time:
        return None
    stamp = pd.to_datetime(buy_time, utc=True, format="ISO8601")
    return stamp.tz_convert(_NY).date().isoformat()


def _eligible_buys(position: dict) -> list[dict]:
    """Filter original entries in Chicago time, including DST and split exits."""
    buys = position.get("_buy_allocations", [position])
    eligible = []
    for buy in buys:
        if not buy.get("buy_time") or (buy.get("shares") or 0) <= 0:
            continue
        stamp = pd.to_datetime(buy["buy_time"], utc=True, format="ISO8601")
        if stamp.tz_convert(_CHICAGO).time() < HOLD_ENTRY_CUTOFF:
            eligible.append(buy)
    return eligible


def _bar_on(history: pd.DataFrame, day: str) -> pd.Series | None:
    if history.empty:
        return None
    day_ts = pd.Timestamp(day)
    index = pd.to_datetime(history.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(_NY).tz_localize(None)
    index = index.normalize()
    matched = history.loc[index == day_ts.normalize()]
    if matched.empty:
        return None
    return matched.iloc[-1]


def _float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def session_open_close(ticker: str, day: str, price_store: PriceStore) -> tuple[float | None, float | None]:
    try:
        history = price_store.read(ticker)
    except FileNotFoundError:
        return None, None
    bar = _bar_on(history, day)
    if bar is None:
        return None, None
    open_px = _float_or_none(bar["Open"]) if "Open" in bar.index else None
    close_px = _float_or_none(bar["Close"]) if "Close" in bar.index else None
    return open_px, close_px


def apply_hold_to_close(
    positions: list[dict],
    price_store: PriceStore | None = None,
    completed_through: date | None = None,
    live_quotes: dict[str, dict] | None = None,
) -> list[dict]:
    """Add 8:40 CT Open / 2:55 CT Close P&L on each lot.

    Only pre-9 AM CT buys count; prices from each buy session's Open and Close,
    not the actual fill. In-progress sessions stay None. After the bell,
    if today's bar is not on disk yet, `live_quotes` (open/day close) fills
    the same cells until nightly writes the bar. Close remains visible even
    for excluded lots; only the counterfactual P&L and its capital are filtered.
    """
    store = price_store if price_store is not None else PriceStore()
    if completed_through is None:
        from stock_picker.ingestion.session import last_completed_session_date

        cutoff = last_completed_session_date()
    else:
        cutoff = completed_through
    quotes = live_quotes or {}
    cache: dict[tuple[str, str], tuple[float | None, float | None]] = {}

    def prices_for(ticker: str, buy_day: str) -> tuple[float | None, float | None]:
        if date.fromisoformat(buy_day) > cutoff:
            return None, None
        key = (ticker, buy_day)
        if key not in cache:
            open_px, close_px = session_open_close(ticker, buy_day, store)
            if (open_px is None or close_px is None) and buy_day == cutoff.isoformat():
                quote = quotes.get(ticker) or {}
                live_open, live_last = quote.get("open"), quote.get("last")
                if live_open and live_last:
                    open_px, close_px = float(live_open), float(live_last)
            cache[key] = (open_px, close_px)
        return cache[key]

    annotated = []
    for position in positions:
        row = dict(position)
        row.pop("_buy_allocations", None)
        row["hold_open_price"] = None
        row["hold_close_price"] = None
        row["hold_close_pnl"] = None
        eligible = _eligible_buys(position)
        row["hold_eligible_shares"] = sum(buy["shares"] for buy in eligible)
        row["hold_eligible_invested"] = round(
            sum(buy.get("invested", 0) for buy in eligible), NOTIONAL_DECIMAL_PLACES
        )
        buy_day = _buy_session_day(position.get("buy_time"))
        shares = position.get("shares") or 0.0
        if not buy_day or shares <= 0:
            annotated.append(row)
            continue
        open_px, close_px = prices_for(position["ticker"], buy_day)
        row["hold_open_price"] = round(open_px, NOTIONAL_DECIMAL_PLACES) if open_px is not None else None
        row["hold_close_price"] = round(close_px, NOTIONAL_DECIMAL_PLACES) if close_px is not None else None
        pnl = 0.0
        for buy in eligible:
            entry_day = _buy_session_day(buy["buy_time"])
            entry_open, entry_close = prices_for(position["ticker"], entry_day)
            if entry_open is None or entry_close is None:
                break  # Never publish a partial result for an incompletely priced lot.
            pnl += (entry_close - entry_open) * buy["shares"]
        else:
            if eligible:
                row["hold_close_pnl"] = round(pnl, NOTIONAL_DECIMAL_PLACES)
        annotated.append(row)
    return annotated
