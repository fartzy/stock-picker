"""Observed next-open prices for an on-demand What If hold comparison.

This is hindsight, not an overnight-model prediction. Both prices come from
one verified raw Massive series (or a session-dated current snapshot), and a
corporate action on the next session makes the simple price-only comparison
unavailable rather than silently misrepresenting a shareholder return.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isclose
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.ingestion.massive_overnight import (
    MassiveOvernightClient,
    MassiveOvernightError,
    TICKER_PATTERN,
)
from stock_picker.training.overnight_dataset import next_expected_session


@dataclass(frozen=True)
class ObservedNextOpen:
    ticker: str
    session: date
    next_session: date | None
    status: str
    verified_close: float | None = None
    next_open: float | None = None
    reason: str | None = None


def observed_next_open(
    ticker: str,
    session: date,
    *,
    client: MassiveOvernightClient | None = None,
    today: date | None = None,
) -> ObservedNextOpen:
    """Return an actual next open only after it is observed and action-checked."""
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("invalid ticker")
    next_session = next_expected_session(session)
    if next_session is None:
        return ObservedNextOpen(ticker, session, None, "unavailable", reason="exchange session unavailable")
    current_date = today or datetime.now(ZoneInfo("America/New_York")).date()
    if next_session > current_date:
        return ObservedNextOpen(ticker, session, next_session, "awaiting_next_open")

    provider = client or MassiveOvernightClient()
    try:
        verified = provider.fetch(ticker, session, session if next_session == current_date else next_session)
        session_stamp = pd.Timestamp(session)
        next_stamp = pd.Timestamp(next_session)
        if session_stamp not in verified.history.index:
            return ObservedNextOpen(ticker, session, next_session, "unavailable", reason="verified close missing")
        close = float(verified.history.loc[session_stamp, "Close"])
        if next_session == current_date:
            current = provider.fetch_current_open(ticker, next_session)
            if current.corporate_action != "verified_none":
                return ObservedNextOpen(ticker, session, next_session, "corporate_action", reason=current.corporate_action)
            if not isclose(close, current.previous_close, rel_tol=1e-4, abs_tol=1e-4):
                return ObservedNextOpen(ticker, session, next_session, "unavailable", reason="raw close and snapshot previous close disagree")
            opened = current.open
        else:
            if next_stamp not in verified.history.index:
                return ObservedNextOpen(ticker, session, next_session, "unavailable", reason="verified next open missing")
            action = verified.provenance.loc[next_stamp, "corporate_action"]
            if action != "verified_none":
                return ObservedNextOpen(ticker, session, next_session, "corporate_action", reason=str(action))
            opened = float(verified.history.loc[next_stamp, "Open"])
        return ObservedNextOpen(ticker, session, next_session, "observed", close, opened)
    except MassiveOvernightError as exc:
        if next_session == current_date and str(exc) == "single-ticker snapshot is not dated to the scenario session":
            return ObservedNextOpen(ticker, session, next_session, "awaiting_next_open")
        return ObservedNextOpen(ticker, session, next_session, "unavailable", reason=str(exc))
