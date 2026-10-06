import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from zoneinfo import ZoneInfo

from stock_picker.ingestion import polygon_stream


def _ms(hour: int, minute: int, second: int = 0) -> int:
    return int(
        datetime(2026, 10, 6, hour, minute, second, tzinfo=ZoneInfo("America/New_York")).timestamp()
        * 1000
    )


def test_second_aggregate_uses_official_op_not_second_bar_open():
    event = {"ev": "A", "sym": "AAPL", "s": _ms(9, 30, 6), "op": 101.25, "o": 102, "c": 101.5}

    assert polygon_stream.quote_from_second_aggregate(
        event, date(2026, 10, 6), {"AAPL": 100}
    ) == ("AAPL", {"open": 101.25, "last": 101.5, "prev_close": 100})


def test_second_aggregate_rejects_premarket_and_missing_official_open():
    prev = {"AAPL": 100}

    assert polygon_stream.quote_from_second_aggregate(
        {"ev": "A", "sym": "AAPL", "s": _ms(9, 29, 59), "op": 101},
        date(2026, 10, 6), prev,
    ) is None
    assert polygon_stream.quote_from_second_aggregate(
        {"ev": "A", "sym": "AAPL", "s": _ms(9, 30, 1), "o": 101},
        date(2026, 10, 6), prev,
    ) is None
    assert polygon_stream.quote_from_second_aggregate(
        {"ev": "A", "sym": "AAPL", "s": _ms(9, 30, 1), "op": 101},
        date(2026, 10, 5), prev,
    ) is None


def test_previous_close_snapshot_keeps_universe_class_share_name():
    raw = [{"ticker": "BRK.B", "prevDay": {"c": 498.2}}, {"ticker": "AAPL", "prevDay": {"c": 0}}]

    assert polygon_stream.previous_closes_from_snapshot(raw, {"BRK-B", "AAPL"}) == {
        "BRK-B": 498.2
    }


def test_stream_authenticates_subscribes_and_returns_official_open(monkeypatch):
    messages = iter([
        json.dumps([{"ev": "status", "status": "connected"}]),
        json.dumps([{"ev": "status", "status": "auth_success"}]),
        json.dumps([{"ev": "status", "status": "success"}]),
        json.dumps([{"ev": "A", "sym": "AAPL", "s": _ms(9, 30, 2), "op": 101.25, "c": 101.4}]),
    ])

    class Socket:
        def __init__(self):
            self.sent = []

        async def recv(self):
            return next(messages)

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    socket = Socket()

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield socket

    monkeypatch.setattr(polygon_stream, "connect", fake_connect)
    monkeypatch.setattr(
        polygon_stream, "fetch_polygon_snapshot",
        lambda key: [{"ticker": "AAPL", "prevDay": {"c": 100}}],
    )

    quotes = asyncio.run(
        polygon_stream._collect_opens("secret", {"AAPL"}, date(2026, 10, 6), time.monotonic() + 10)
    )

    assert quotes == {"AAPL": {"open": 101.25, "last": 101.4, "prev_close": 100}}
    assert socket.sent == [
        {"action": "auth", "params": "secret"},
        {"action": "subscribe", "params": "A.*"},
    ]


def test_stream_does_not_accept_a_delayed_entitlement(monkeypatch):
    messages = iter([
        json.dumps([{"ev": "status", "status": "connected"}]),
        json.dumps([{"ev": "status", "status": "auth_success"}]),
        json.dumps([{"ev": "status", "status": "error", "message": "real-time unavailable"}]),
    ])

    class Socket:
        async def recv(self):
            return next(messages)

        async def send(self, payload):
            pass

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield Socket()

    def fail_if_snapshot_requested(key):
        raise AssertionError("no snapshot after entitlement denial")

    monkeypatch.setattr(polygon_stream, "connect", fake_connect)
    monkeypatch.setattr(polygon_stream, "fetch_polygon_snapshot", fail_if_snapshot_requested)

    assert asyncio.run(
        polygon_stream._collect_opens("secret", {"AAPL"}, date(2026, 10, 6), time.monotonic() + 10)
    ) == {}
