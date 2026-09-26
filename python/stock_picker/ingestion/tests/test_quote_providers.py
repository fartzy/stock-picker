from datetime import date
from unittest.mock import patch

from stock_picker.ingestion.quote_providers import (
    YahooQuoteProvider,
    fill_quotes,
    fetch_quotes,
    provider_names_from_env,
    quote_providers,
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
