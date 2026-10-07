"""Observed next-open What If comparisons never use a future or mixed bar."""

from datetime import date, datetime, timezone

import pandas as pd

from stock_picker.ingestion.massive_overnight import MassiveOvernightError, VerifiedCurrentOpen, VerifiedOvernightBars
from stock_picker.training.overnight_hold import observed_next_open


def verified(*, action="verified_none", next_bar=True):
    sessions = ["2026-10-05", "2026-10-06"] if next_bar else ["2026-10-05"]
    frame = pd.DataFrame(
        {"Open": [10.0, 10.4][:len(sessions)], "Close": [10.2, 10.5][:len(sessions)]},
        index=pd.DatetimeIndex(sessions),
    )
    provenance = pd.DataFrame(
        {"corporate_action": ["verified_none", action][:len(sessions)]},
        index=frame.index,
    )
    return VerifiedOvernightBars(frame, provenance, (), ())


class FakeClient:
    def __init__(self, bars, current=None):
        self.bars = bars
        self.current = current
        self.fetch_calls = []

    def fetch(self, ticker, start, end):
        self.fetch_calls.append((ticker, start, end))
        return self.bars

    def fetch_current_open(self, ticker, session):
        return self.current


def test_historical_actual_next_open_comes_from_same_verified_raw_frame():
    client = FakeClient(verified())
    result = observed_next_open("WERN", date(2026, 10, 5), client=client, today=date(2026, 10, 7))
    assert (result.status, result.verified_close, result.next_open) == ("observed", 10.2, 10.4)
    assert client.fetch_calls == [("WERN", date(2026, 10, 5), date(2026, 10, 6))]


def test_future_next_open_does_not_call_provider():
    client = FakeClient(verified())
    result = observed_next_open("WERN", date(2026, 10, 6), client=client, today=date(2026, 10, 6))
    assert result.status == "awaiting_next_open"
    assert client.fetch_calls == []


def test_current_session_snapshot_is_checked_against_previous_raw_close():
    now = datetime(2026, 10, 6, 14, tzinfo=timezone.utc)
    current = VerifiedCurrentOpen(10.4, 10.2, 10.5, now, now, "verified_none")
    client = FakeClient(verified(next_bar=False), current)
    result = observed_next_open("WERN", date(2026, 10, 5), client=client, today=date(2026, 10, 6))
    assert result.status == "observed"
    assert result.next_open == 10.4
    client.current = VerifiedCurrentOpen(10.4, 9.2, 10.5, now, now, "verified_none")
    assert observed_next_open("WERN", date(2026, 10, 5), client=client, today=date(2026, 10, 6)).status == "unavailable"


def test_next_session_corporate_action_is_not_plain_price_profit():
    client = FakeClient(verified(action="dividend"))
    result = observed_next_open("WERN", date(2026, 10, 5), client=client, today=date(2026, 10, 7))
    assert result.status == "corporate_action"
    assert result.next_open is None


def test_stale_current_snapshot_means_next_open_is_still_awaited():
    client = FakeClient(verified(next_bar=False))
    def stale(ticker, session):
        raise MassiveOvernightError("single-ticker snapshot is not dated to the scenario session")
    client.fetch_current_open = stale
    result = observed_next_open("WERN", date(2026, 10, 5), client=client, today=date(2026, 10, 6))
    assert result.status == "awaiting_next_open"
