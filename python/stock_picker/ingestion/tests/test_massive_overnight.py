"""Provider responses must prove raw basis and complete action screening."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_picker.ingestion.massive_overnight import (
    MassiveOvernightClient,
    MassiveOvernightError,
    parse_action_page,
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
    today = datetime.now(ZoneInfo("America/New_York")).date()
    session = FakeSession([])
    with pytest.raises(MassiveOvernightError, match="completed historical"):
        MassiveOvernightClient("test-key", session=session).fetch("AAPL", today, today)
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
