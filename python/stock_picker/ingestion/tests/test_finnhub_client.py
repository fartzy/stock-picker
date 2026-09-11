from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from stock_picker.ingestion.finnhub_client import (
    earnings_tickers_from_calendar,
    fetch_finnhub_quotes,
    finnhub_api_key,
    finnhub_symbol,
    quote_from_finnhub_payload,
)
from stock_picker.ingestion.yfinance_client import fetch_quotes


def _unix(year, month, day, hour=9, minute=31):
    dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    return int(dt.timestamp())


def test_finnhub_symbol_maps_yahoo_class_shares():
    assert finnhub_symbol("BRK-B") == "BRK.B"
    assert finnhub_symbol("AAPL") == "AAPL"


def test_quote_from_finnhub_payload_uses_day_open_not_last():
    payload = {"o": 103.47, "c": 96.07, "pc": 102.48, "t": _unix(2026, 9, 10, 9, 31)}

    quote = quote_from_finnhub_payload(payload, as_of=date(2026, 9, 10))

    assert quote == {"open": 103.47, "last": 96.07, "prev_close": 102.48}


def test_quote_from_finnhub_payload_drops_a_stale_quote():
    payload = {"o": 84.33, "c": 81.28, "pc": 85.30, "t": _unix(2026, 9, 8, 16, 0)}

    assert quote_from_finnhub_payload(payload, as_of=date(2026, 9, 10)) is None


def test_quote_from_finnhub_payload_drops_a_zero_open():
    payload = {"o": 0, "c": 96.07, "pc": 102.48, "t": _unix(2026, 9, 10, 9, 31)}

    assert quote_from_finnhub_payload(payload, as_of=date(2026, 9, 10)) is None


def test_finnhub_api_key_reads_email_colon_key(tmp_path):
    key_file = tmp_path / "finnhub.txt"
    key_file.write_text("user@example.com: test-token\n")

    with patch.dict("os.environ", {}, clear=True):
        assert finnhub_api_key(key_file=key_file) == "test-token"


def test_finnhub_api_key_env_wins(tmp_path):
    key_file = tmp_path / "finnhub.txt"
    key_file.write_text("user@example.com: from-file\n")

    with patch.dict("os.environ", {"FINNHUB_API_KEY": "from-env"}):
        assert finnhub_api_key(key_file=key_file) == "from-env"


def test_fetch_finnhub_quotes_skips_when_leftovers_exceed_the_cap():
    tickers = [f"T{i}" for i in range(41)]

    with patch("stock_picker.ingestion.finnhub_client.fetch_one_finnhub_quote") as mock_fetch:
        quotes = fetch_finnhub_quotes(tickers, as_of=date(2026, 9, 10), api_key="k", sleep_seconds=0)

    assert quotes == {}
    mock_fetch.assert_not_called()


def test_fetch_quotes_uses_finnhub_only_for_yahoo_leftovers():
    yahoo = {
        "AAPL": {"open": 316.79, "last": 319.0, "prev_close": 315.34},
    }
    finnhub_payload = {"o": 103.47, "c": 96.07, "pc": 102.48, "t": _unix(2026, 9, 10, 9, 31)}

    with (
        patch("stock_picker.ingestion.polygon_client.fetch_polygon_quotes", return_value={}),
        patch("stock_picker.ingestion.yfinance_client.fetch_yahoo_quotes", return_value=yahoo),
        patch("stock_picker.ingestion.finnhub_client.finnhub_api_key", return_value="k"),
        patch(
            "stock_picker.ingestion.finnhub_client.fetch_one_finnhub_quote",
            return_value=finnhub_payload,
        ) as mock_one,
    ):
        quotes = fetch_quotes(["AAPL", "SIG"], as_of=date(2026, 9, 10))

    assert quotes["AAPL"]["open"] == 316.79
    assert quotes["SIG"]["open"] == 103.47
    mock_one.assert_called_once()
    assert mock_one.call_args.args[0] == "SIG"


def test_earnings_tickers_from_calendar_maps_finnhub_dots_to_yahoo_dashes():
    payload = {
        "earningsCalendar": [
            {"symbol": "BRK.B", "date": "2026-09-10"},
            {"symbol": "AAPL", "date": "2026-09-09"},
            {"symbol": "MSFT", "date": "2026-09-10"},
        ]
    }

    hits = earnings_tickers_from_calendar(
        payload, wanted={"BRK-B", "AAPL", "MSFT"}, as_of=date(2026, 9, 10)
    )

    assert hits == {"BRK-B", "MSFT"}
