"""US cash-session calendar: which daily bar is a completed close.

Yahoo's 1d pull during the session includes today's in-progress candle.
Features and training must stop at the last *finished* session -- after
16:00 ET that is today; before the bell it is the previous weekday.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

US_EASTERN = ZoneInfo("America/New_York")
# Regular session ends 16:00 ET / 15:00 CT. Yahoo's official daily Close
# usually exists within a few minutes. The 3:30 CT nightly job is 30 minutes
# after the bell -- well past this settle.
SESSION_CLOSE_HOUR = 16
SESSION_CLOSE_SETTLE_MINUTES = 15


def last_completed_session_date(now: datetime | None = None) -> date:
    """Most recent weekday whose regular session has closed (plus settle)."""
    clock = now.astimezone(US_EASTERN) if now is not None else datetime.now(US_EASTERN)
    closed = clock.hour > SESSION_CLOSE_HOUR or (
        clock.hour == SESSION_CLOSE_HOUR and clock.minute >= SESSION_CLOSE_SETTLE_MINUTES
    )
    day = clock.date()
    if not closed:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def completed_sessions(history: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop any bar dated after the last completed US cash session."""
    if history.empty:
        return history
    cutoff = last_completed_session_date(now)
    timestamps = pd.DatetimeIndex(history.index)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("America/New_York").tz_localize(None)
    keep = pd.Index(timestamps.date) <= cutoff
    return history.iloc[keep]
