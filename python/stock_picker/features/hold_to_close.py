"""Counterfactual session: buy 8:40 AM CT, sell 2:55 PM CT.

Not "your fill vs the close". 8:40 CT is 9:40 ET (ten minutes after the
bell); 2:55 CT is 15:55 ET (five minutes before the close). Daily Open and
Close are the stored prints for that window -- we do not refetch 1-minute
bars. Same share count as the lot. In-progress today stays blank.
"""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.storage.price_store import PriceStore

_NY = ZoneInfo("America/New_York")
NOTIONAL_DECIMAL_PLACES = 2


def _buy_session_day(buy_time: str | None) -> str | None:
    if not buy_time:
        return None
    stamp = pd.to_datetime(buy_time, utc=True, format="ISO8601")
    return stamp.tz_convert(_NY).date().isoformat()


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
) -> list[dict]:
    """Add 8:40 CT Open / 2:55 CT Close P&L on each lot.

    Share count from the lot; prices from that session's Open and Close,
    not the actual fill. In-progress sessions stay None.
    """
    store = price_store if price_store is not None else PriceStore()
    if completed_through is None:
        from stock_picker.ingestion.session import last_completed_session_date

        cutoff = last_completed_session_date()
    else:
        cutoff = completed_through
    cache: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    annotated = []
    for position in positions:
        row = dict(position)
        row["hold_open_price"] = None
        row["hold_close_price"] = None
        row["hold_close_pnl"] = None
        buy_day = _buy_session_day(position.get("buy_time"))
        shares = position.get("shares") or 0.0
        if not buy_day or shares <= 0:
            annotated.append(row)
            continue
        if cutoff is not None and date.fromisoformat(buy_day) > cutoff:
            annotated.append(row)
            continue
        key = (position["ticker"], buy_day)
        if key not in cache:
            cache[key] = session_open_close(position["ticker"], buy_day, store)
        open_px, close_px = cache[key]
        if open_px is None or close_px is None:
            annotated.append(row)
            continue
        row["hold_open_price"] = round(open_px, NOTIONAL_DECIMAL_PLACES)
        row["hold_close_price"] = round(close_px, NOTIONAL_DECIMAL_PLACES)
        row["hold_close_pnl"] = round((close_px - open_px) * shares, NOTIONAL_DECIMAL_PLACES)
        annotated.append(row)
    return annotated
