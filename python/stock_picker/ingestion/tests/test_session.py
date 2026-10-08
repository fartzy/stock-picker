from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.ingestion.session import (
    cash_session_has_ended,
    cash_session_is_open,
    cash_session_window,
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


def test_display_close_starts_at_bell_without_advancing_training_cutoff():
    before = datetime(2026, 9, 24, 15, 59, tzinfo=ET)
    at_bell = datetime(2026, 9, 24, 16, 0, tzinfo=ET)
    assert cash_session_has_ended(before) is False
    assert cash_session_has_ended(at_bell) is True
    assert session_has_closed(at_bell) is False


def test_cash_session_open_uses_weekday_eastern_open_and_close_boundaries():
    assert cash_session_is_open(datetime(2026, 10, 6, 9, 29, tzinfo=ET)) is False
    assert cash_session_is_open(datetime(2026, 10, 6, 9, 30, tzinfo=ET)) is True
    assert cash_session_is_open(datetime(2026, 10, 6, 15, 55, tzinfo=ET)) is True
    assert cash_session_is_open(datetime(2026, 10, 6, 16, 0, tzinfo=ET)) is False
    assert cash_session_is_open(datetime(2026, 10, 6, 16, 5, tzinfo=ET)) is False
    assert cash_session_is_open(datetime(2026, 10, 10, 15, 55, tzinfo=ET)) is False


def test_xnys_early_close_and_holiday_boundaries():
    early = datetime(2026, 11, 27, 12, 55, tzinfo=ET)
    window = cash_session_window(early)
    assert window is not None
    assert window.close_at.astimezone(ET) == datetime(2026, 11, 27, 13, 0, tzinfo=ET)
    assert cash_session_is_open(early) is True
    assert cash_session_is_open(datetime(2026, 11, 27, 13, 0, tzinfo=ET)) is False
    assert cash_session_is_open(datetime(2026, 11, 27, 13, 5, tzinfo=ET)) is False
    assert cash_session_window(datetime(2026, 11, 26, 12, 0, tzinfo=ET)) is None


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
