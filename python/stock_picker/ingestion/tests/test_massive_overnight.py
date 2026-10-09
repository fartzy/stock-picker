"""Provider responses must prove raw basis and complete action screening."""

from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_picker.ingestion.massive_overnight import (
    MassiveOvernightClient,
    MassiveOvernightError,
    parse_action_page,
    parse_current_snapshot,
    parse_current_trade,
    parse_daily_bar_page,
)

START = date(2026, 10, 1)
END = date(2026, 10, 2)
BAR_PATH = "/v2/aggs/ticker/AAPL/range/1/day/2026-10-01/2026-10-02"


def bar(day: str):
    return {
        "t": int(pd.Timestamp(day, tz="America/New_York").timestamp() * 1000),
        "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 1000,
    }


def bars_page(*items, adjusted=False, next_url=None):
    return {
        "status": "OK", "ticker": "AAPL", "adjusted": adjusted,
        "results": list(items), "resultsCount": len(items), "next_url": next_url,
    }


def actions_page(*items, next_url=None):
    return {"status": "OK", "results": list(items), "next_url": next_url}


def snapshot(day=START, *, ticker="AAPL", opened=10.0):
    stamped = int(pd.Timestamp(day, tz="America/New_York").replace(hour=15).timestamp() * 1_000_000_000)
    return {
        "status": "OK",
        "ticker": {
            "ticker": ticker, "updated": stamped,
            "day": {"o": opened}, "prevDay": {"c": 9.9},
            "lastTrade": {"p": 10.2, "t": stamped},
        },
    }


def test_current_open_requires_session_dated_snapshot_and_completed_action_checks():
    fake = FakeSession([
        FakeResponse(snapshot()), FakeResponse(actions_page()), FakeResponse(actions_page()),
    ])
    observed = MassiveOvernightClient("test-key", session=fake).fetch_current_open("AAPL", START)
    assert observed.open == 10.0
    assert observed.previous_close == 9.9
    assert observed.last_trade == 10.2
    assert observed.corporate_action == "verified_none"
    assert fake.calls[0][0].endswith("/v2/snapshot/locale/us/markets/stocks/tickers/AAPL")
    assert fake.calls[1][1]["execution_date.gte"] == START.isoformat()


def test_current_trade_uses_one_snapshot_and_preserves_trade_timestamp():
    fake = FakeSession([FakeResponse(snapshot())])
    observed = MassiveOvernightClient("test-key", session=fake).fetch_current_trade("AAPL", START)
    assert observed.price == 10.2
    assert observed.observed_at.date() == START
    assert len(fake.calls) == 1


@pytest.mark.parametrize("payload", [
    snapshot(END), snapshot(ticker="MSFT"),
    {**snapshot(), "ticker": {**snapshot()["ticker"], "lastTrade": {"p": 10.2}}},
    {**snapshot(), "ticker": {**snapshot()["ticker"], "lastTrade": {"p": 0, "t": snapshot()["ticker"]["lastTrade"]["t"]}}},
])
def test_current_trade_rejects_wrong_session_ticker_or_missing_trade(payload):
    with pytest.raises(MassiveOvernightError):
        parse_current_trade(payload, "AAPL", START)


@pytest.mark.parametrize("payload", [
    snapshot(END), snapshot(ticker="MSFT"), snapshot(opened=0),
    {**snapshot(), "status": "ERROR"},
])
def test_current_snapshot_rejects_stale_wrong_or_invalid_open(payload):
    with pytest.raises(MassiveOvernightError):
        parse_current_snapshot(payload, "AAPL", START)


def test_current_snapshot_names_missing_regular_session_open():
    payload = snapshot(opened=0)
    with pytest.raises(MassiveOvernightError, match="no current-session open"):
        parse_current_snapshot(payload, "AAPL", START)


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return next(self.responses)


def test_complete_pages_mark_only_checked_dates_and_preserve_events():
    second_split_page = "https://api.massive.com/stocks/v1/splits?cursor=second"
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"), bar("2026-10-02"))),
        FakeResponse(actions_page(next_url=second_split_page)),
        FakeResponse(actions_page()),
        FakeResponse(actions_page({"ticker": "AAPL", "execution_date": "2026-10-02", "id": "split-1"})),
        FakeResponse(actions_page({"ticker": "AAPL", "ex_dividend_date": "2026-10-02", "id": "div-1"})),
    ])
    result = MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, END)

    assert result.history.index.tolist() == [pd.Timestamp(START), pd.Timestamp(END)]
    assert result.provenance.loc["2026-10-01", "corporate_action"] == "verified_none"
    assert result.provenance.loc["2026-10-02", "corporate_action"] == "split"
    assert result.provenance.loc["2026-10-02", "action_event_ids"] == "split:split-1|dividend:div-1"
    assert result.provenance["basis"].eq("raw").all()
    assert len(session.calls) == 5
    assert session.calls[0][1]["adjusted"] == "false"
    assert session.calls[1][1]["execution_date.gte"] == START.isoformat()
    assert session.calls[1][1]["execution_date.lte"] == END.isoformat()
    assert session.calls[2][0] == "https://api.massive.com/stocks/v1/splits"
    assert session.calls[2][1]["execution_date.gte"] == START.isoformat()
    assert session.calls[2][1]["execution_date.lte"] == START.isoformat()
    assert session.calls[3][1]["execution_date.gte"] == END.isoformat()
    assert session.calls[4][1]["ex_dividend_date.gte"] == START.isoformat()


def test_aggregate_cursor_can_advance_start_as_timestamp():
    cursor = "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/1791000000000/2026-10-02?cursor=second"
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"), next_url=cursor)),
        FakeResponse(bars_page(bar("2026-10-02"))),
        FakeResponse(actions_page()),
        FakeResponse(actions_page()),
    ])
    result = MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, END)
    assert len(result.history) == 2
    assert session.calls[1][0] == cursor


def test_incomplete_action_pagination_cannot_certify_clean_dates():
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"))),
        FakeResponse(actions_page(next_url="https://api.massive.com/stocks/v1/splits?cursor=next")),
    ])
    with pytest.raises(MassiveOvernightError, match="request limit"):
        MassiveOvernightClient("test-key", session=session, max_pages=1).fetch("AAPL", START, END)


def test_one_day_action_pagination_fails_closed():
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"))),
        FakeResponse(actions_page(next_url="https://api.massive.com/stocks/v1/splits?cursor=next")),
    ])
    with pytest.raises(MassiveOvernightError, match="one-day action"):
        MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, START)
    assert len(session.calls) == 2


def test_action_cursor_with_scope_parameters_is_rejected():
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"))),
        FakeResponse(actions_page(next_url="https://api.massive.com/stocks/v1/splits?cursor=next&ticker=MSFT")),
    ])
    with pytest.raises(MassiveOvernightError, match="cursor-only"):
        MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, END)
    assert len(session.calls) == 2


def test_old_range_is_rejected_before_any_request():
    session = FakeSession([])
    with pytest.raises(MassiveOvernightError, match="coverage"):
        MassiveOvernightClient("test-key", session=session).fetch(
            "AAPL", date(2020, 1, 1), date(2020, 1, 3)
        )
    assert not session.calls


def test_more_restrictive_action_coverage_is_enforced():
    session = FakeSession([])
    with pytest.raises(MassiveOvernightError, match="coverage"):
        MassiveOvernightClient(
            "test-key", session=session, action_coverage_start=END + timedelta(days=1)
        ).fetch("AAPL", START, END)
    assert not session.calls


def test_current_session_is_not_certified_as_complete():
    now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("America/New_York"))
    session = FakeSession([])
    with patch("stock_picker.ingestion.massive_overnight.datetime", wraps=datetime) as clock:
        clock.now.return_value = now
        with pytest.raises(MassiveOvernightError, match="completed historical"):
            MassiveOvernightClient("test-key", session=session).fetch("AAPL", now.date(), now.date())
    assert not session.calls


def test_empty_bar_cursor_page_fails_closed():
    cursor = "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/1791000000000/2026-10-02?cursor=second"
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"), next_url=cursor)),
        FakeResponse(bars_page()),
    ])
    with pytest.raises(MassiveOvernightError, match="empty cursor page"):
        MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, END)
    assert len(session.calls) == 2


def test_cross_host_action_cursor_is_rejected_before_authentication():
    session = FakeSession([
        FakeResponse(bars_page(bar("2026-10-01"))),
        FakeResponse(actions_page(next_url="https://elsewhere.example/stocks/v1/splits?cursor=next")),
    ])
    with pytest.raises(MassiveOvernightError, match="pagination link"):
        MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, END)
    assert len(session.calls) == 2


@pytest.mark.parametrize("payload", [
    bars_page(bar("2026-10-01"), adjusted=True),
    {**bars_page(bar("2026-10-01")), "ticker": "MSFT"},
    {**bars_page(bar("2026-10-01")), "resultsCount": 2},
    bars_page(bar("2026-09-30")),
    bars_page({**bar("2026-10-01"), "o": 0}),
    bars_page({**bar("2026-10-01"), "o": 10**1000}),
])
def test_bar_parser_rejects_unverified_basis_or_invalid_rows(payload):
    with pytest.raises(MassiveOvernightError):
        parse_daily_bar_page(payload, "AAPL", START, END)


def test_action_parser_rejects_missing_date_or_wrong_ticker():
    for record in (
        {"ticker": "AAPL", "id": "missing-date"},
        {"ticker": "MSFT", "execution_date": "2026-10-02"},
        {"ticker": "AAPL", "execution_date": "2026-09-30"},
    ):
        with pytest.raises(MassiveOvernightError):
            parse_action_page(actions_page(record), "AAPL", START, END, kind="split")


def test_http_failure_stops_before_action_verification():
    response = FakeResponse(bars_page(bar("2026-10-01")))
    response.status_code = 403
    session = FakeSession([response])
    with pytest.raises(MassiveOvernightError, match="HTTP 403"):
        MassiveOvernightClient("test-key", session=session).fetch("AAPL", START, END)
    assert len(session.calls) == 1
