from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from stock_picker.ingestion.quote_providers import (
    YahooQuoteProvider,
    fill_quotes,
    fetch_morning_quotes,
    fetch_quotes,
    provider_names_from_env,
    quote_providers,
    wait_for_morning_snapshot,
)


class _Stub:
    def __init__(self, name: str, payload: dict[str, dict]):
        self.name = name
        self.payload = payload
        self.called_with: list[str] | None = None

    def fetch(self, tickers: list[str], as_of: date) -> dict[str, dict]:
        self.called_with = list(tickers)
        return {ticker: self.payload[ticker] for ticker in tickers if ticker in self.payload}


def test_fill_quotes_stops_after_the_first_provider_has_everyone():
    first = _Stub("polygon", {"WRBY": {"open": 26.27, "last": 26.71, "prev_close": 26.27}})
    second = _Stub("yahoo", {"WRBY": {"open": 23.18, "last": 25.81, "prev_close": 26.27}})

    quotes = fill_quotes(["WRBY"], date(2026, 9, 25), [first, second])

    assert quotes["WRBY"]["open"] == 26.27
    assert second.called_with is None


def test_fill_quotes_asks_the_next_provider_only_for_missing_names():
    first = _Stub("polygon", {"PS": {"open": 52.96, "last": 56.16, "prev_close": 52.28}})
    second = _Stub("yahoo", {"WRBY": {"open": 26.27, "last": 26.71, "prev_close": 26.27}})

    quotes = fill_quotes(["WRBY", "PS"], date(2026, 9, 25), [first, second])

    assert quotes["PS"]["open"] == 52.96
    assert quotes["WRBY"]["open"] == 26.27
    assert second.called_with == ["WRBY"]


def test_fill_quotes_keeps_polygon_open_even_when_it_equals_yesterday():
    """A flat open (WRBY 26.27 = Thursday close) is a real print from Polygon."""
    polygon = _Stub(
        "polygon",
        {"WRBY": {"open": 26.27, "last": 26.71, "prev_close": 26.27}},
    )

    quotes = fill_quotes(["WRBY"], date(2026, 9, 25), [polygon])

    assert quotes["WRBY"]["open"] == 26.27


def test_provider_names_from_env_default_is_polygon_then_yahoo():
    assert provider_names_from_env("") == ("polygon", "yahoo", "finnhub")
    assert provider_names_from_env("yahoo,polygon") == ("yahoo", "polygon")


def test_quote_providers_skips_unknown_names():
    names = [provider.name for provider in quote_providers(("polygon", "nope", "yahoo"))]
    assert names == ["polygon", "yahoo"]


def test_fetch_quotes_uses_injected_providers():
    stub = _Stub("polygon", {"AAPL": {"open": 100.0, "last": 101.0, "prev_close": 99.0}})

    quotes = fetch_quotes(["AAPL"], as_of=date(2026, 9, 25), providers=[stub])

    assert quotes["AAPL"]["open"] == 100.0
    assert stub.called_with == ["AAPL"]


def test_yahoo_provider_drops_a_copied_yesterday_open():
    snapshot = {
        "WRBY": {"open": 23.18, "last": 26.71, "prev_close": 26.27},
        "PS": {"open": 52.96, "last": 56.16, "prev_close": 52.28},
    }
    with (
        patch(
            "stock_picker.ingestion.yfinance_client.fetch_yahoo_quotes",
            return_value=snapshot,
        ),
        patch(
            "stock_picker.ingestion.yfinance_client.prior_session_opens",
            return_value={"WRBY": 23.18, "PS": 49.32},
        ),
    ):
        quotes = YahooQuoteProvider().fetch(["WRBY", "PS"], date(2026, 9, 25))

    assert "WRBY" not in quotes
    assert quotes["PS"]["open"] == 52.96


class _Clock:
    def __init__(self, hour: int, minute: int, second: int):
        self.value = datetime(2026, 9, 30, hour, minute, second, tzinfo=ZoneInfo("America/Chicago"))
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


class _SequencePolygon:
    name = "polygon"

    def __init__(self, clock: _Clock, batches: list[dict[str, dict]]):
        self.clock = clock
        self.batches = iter(batches)
        self.calls: list[tuple[datetime, list[str]]] = []

    def fetch(self, tickers: list[str], as_of: date) -> dict[str, dict]:
        self.calls.append((self.clock.now(), list(tickers)))
        batch = next(self.batches)
        return {ticker: batch[ticker] for ticker in tickers if ticker in batch}


def _quote(open_price: float) -> dict[str, float]:
    return {"open": open_price, "last": open_price, "prev_close": open_price}


def test_morning_snapshots_begin_at_83005_and_retry_missing_every_five_seconds():
    clock = _Clock(8, 29, 58)
    polygon = _SequencePolygon(clock, [{"A": _quote(1)}, {"B": _quote(2)}])
    fallback = _Stub("yahoo", {"A": _quote(10), "B": _quote(20)})

    quotes = fetch_morning_quotes(
        ["A", "B"], providers=[polygon, fallback], now_fn=clock.now, sleep_fn=clock.sleep
    )

    assert quotes == {"A": _quote(1), "B": _quote(2)}
    assert [call[0].strftime("%H:%M:%S") for call in polygon.calls] == ["08:30:05", "08:30:10"]
    assert [call[1] for call in polygon.calls] == [["A", "B"], ["B"]]
    assert fallback.called_with is None


def test_wait_for_morning_snapshot_preserves_830_checkbox_window():
    clock = _Clock(8, 29, 0)

    wait_for_morning_snapshot(now_fn=clock.now, sleep_fn=clock.sleep)

    assert clock.now().strftime("%H:%M:%S") == "08:30:05"
    assert clock.sleeps == [65]


def test_morning_retries_through_833_then_falls_back_only_for_missing():
    clock = _Clock(8, 32, 50)
    polygon = _SequencePolygon(clock, [{"A": _quote(1)}, {}, {}])
    fallback = _Stub("yahoo", {"B": _quote(2)})

    quotes = fetch_morning_quotes(
        ["A", "B"], providers=[polygon, fallback], now_fn=clock.now, sleep_fn=clock.sleep
    )

    assert quotes == {"A": _quote(1), "B": _quote(2)}
    assert [call[0].strftime("%H:%M:%S") for call in polygon.calls] == [
        "08:32:50", "08:32:55", "08:33:00"
    ]
    assert fallback.called_with == ["B"]


def test_morning_scan_after_cutoff_fetches_once_without_sleep():
    clock = _Clock(8, 34, 0)
    polygon = _SequencePolygon(clock, [{}])
    fallback = _Stub("yahoo", {"A": _quote(2)})

    quotes = fetch_morning_quotes(
        ["A"], providers=[polygon, fallback], now_fn=clock.now, sleep_fn=clock.sleep
    )

    assert quotes == {"A": _quote(2)}
    assert len(polygon.calls) == 1
    assert clock.sleeps == []


def test_morning_custom_non_polygon_chain_keeps_single_pass_behavior():
    clock = _Clock(8, 29, 0)
    yahoo = _Stub("yahoo", {"A": _quote(2)})

    quotes = fetch_morning_quotes(
        ["A"], providers=[yahoo], now_fn=clock.now, sleep_fn=clock.sleep
    )

    assert quotes == {"A": _quote(2)}
    assert clock.sleeps == []
