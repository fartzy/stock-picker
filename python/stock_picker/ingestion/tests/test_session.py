from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.ingestion.session import (
    cash_session_date,
    completed_sessions,
    last_completed_session_date,
    session_has_closed,
)

ET = ZoneInfo("America/New_York")


def test_cash_session_date_is_today_on_a_weekday():
    now = datetime(2026, 9, 24, 10, 0, tzinfo=ET)

    assert cash_session_date(now) == datetime(2026, 9, 24).date()
    assert session_has_closed(now) is False


def test_cash_session_date_is_friday_on_the_weekend():
    saturday = datetime(2026, 9, 26, 11, 0, tzinfo=ET)

    assert cash_session_date(saturday) == datetime(2026, 9, 25).date()
    assert session_has_closed(saturday) is True


def test_session_has_closed_after_settle():
    assert session_has_closed(datetime(2026, 9, 24, 16, 20, tzinfo=ET)) is True
    assert session_has_closed(datetime(2026, 9, 24, 15, 59, tzinfo=ET)) is False


def test_before_the_bell_uses_the_previous_weekday():
    # Thursday 08:47 ET -- today's bar is still in progress.
    now = datetime(2026, 9, 10, 8, 47, tzinfo=ET)

    assert last_completed_session_date(now) == datetime(2026, 9, 9).date()


def test_after_settle_uses_today_on_a_weekday():
    now = datetime(2026, 9, 10, 16, 20, tzinfo=ET)

    assert last_completed_session_date(now) == datetime(2026, 9, 10).date()


def test_weekend_rolls_back_to_friday():
    saturday = datetime(2026, 9, 12, 18, 0, tzinfo=ET)
    sunday = datetime(2026, 9, 13, 10, 0, tzinfo=ET)

    assert last_completed_session_date(saturday) == datetime(2026, 9, 11).date()
    assert last_completed_session_date(sunday) == datetime(2026, 9, 11).date()


def test_completed_sessions_drops_an_in_progress_today_bar():
    now = datetime(2026, 9, 11, 8, 47, tzinfo=ET)
    history = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2026-09-09", "2026-09-10", "2026-09-11"]),
    )

    finished = completed_sessions(history, now=now)

    assert list(finished.index.date) == [
        datetime(2026, 9, 9).date(),
        datetime(2026, 9, 10).date(),
    ]
