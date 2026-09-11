from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from stock_picker.ingestion.polygon_client import (
    fetch_polygon_quotes,
    polygon_api_key,
    quotes_from_polygon_snapshot,
)
from stock_picker.ingestion.yfinance_client import fetch_quotes


def _ns(year, month, day, hour=9, minute=31):
    dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    return int(dt.timestamp() * 1_000_000_000)


def test_quotes_from_polygon_snapshot_uses_today_day_open():
    raw = [
        {
            "ticker": "SIG",
            "updated": _ns(2026, 9, 10, 9, 31),
            "day": {"o": 103.47, "c": 96.07},
            "lastTrade": {"p": 96.07, "t": _ns(2026, 9, 10, 9, 31)},
            "prevDay": {"c": 102.48},
        },
        {
            "ticker": "MSFT",
            "updated": _ns(2026, 9, 10, 9, 31),
            "day": {"o": 400.0, "c": 401.0},
            "lastTrade": {"p": 401.0, "t": _ns(2026, 9, 10, 9, 31)},
            "prevDay": {"c": 399.0},
        },
    ]

    quotes = quotes_from_polygon_snapshot(raw, wanted={"SIG"}, as_of=date(2026, 9, 10))

    assert quotes == {"SIG": {"open": 103.47, "last": 96.07, "prev_close": 102.48}}


def test_quotes_from_polygon_snapshot_omits_a_stale_previous_session():
    raw = [
        {
            "ticker": "SIG",
            "updated": _ns(2026, 9, 8, 16, 0),
            "day": {"o": 84.33, "c": 81.28},
            "lastTrade": {"p": 81.28, "t": _ns(2026, 9, 8, 16, 0)},
            "prevDay": {"c": 85.30},
        }
    ]

    quotes = quotes_from_polygon_snapshot(raw, wanted={"SIG"}, as_of=date(2026, 9, 10))

    assert quotes == {}


def test_quotes_from_polygon_snapshot_keeps_a_large_today_open():
    raw = [
        {
            "ticker": "TTAN",
            "updated": _ns(2026, 9, 10, 9, 31),
            "day": {"o": 56.35, "c": 56.40},
            "lastTrade": {"p": 56.40, "t": _ns(2026, 9, 10, 9, 31)},
            "prevDay": {"c": 85.68},
        }
    ]

    quotes = quotes_from_polygon_snapshot(raw, wanted={"TTAN"}, as_of=date(2026, 9, 10))

    assert quotes["TTAN"]["open"] == 56.35


def test_quotes_from_polygon_snapshot_skips_when_day_open_has_not_printed():
    raw = [
        {
            "ticker": "ILLQ",
            "updated": _ns(2026, 9, 10, 9, 30),
            "day": {},
            "lastTrade": {"p": 12.0, "t": _ns(2026, 9, 10, 9, 30)},
            "prevDay": {"c": 11.5},
        }
    ]

    quotes = quotes_from_polygon_snapshot(raw, wanted={"ILLQ"}, as_of=date(2026, 9, 10))

    assert quotes == {}


def test_fetch_polygon_quotes_returns_empty_without_a_key():
    with patch.dict("os.environ", {"BUILD_WORKING_DIRECTORY": "/tmp"}, clear=True):
        assert fetch_polygon_quotes(["SIG"], as_of=date(2026, 9, 10), api_key="") == {}


def test_polygon_api_key_reads_default_key_from_config_file(tmp_path):
    key_file = tmp_path / "polygon-massive.txt"
    key_file.write_text("default_key=test-from-file\ns3_access_key_id=not-the-api-key\n")

    with patch.dict("os.environ", {"BUILD_WORKING_DIRECTORY": str(tmp_path)}, clear=True):
        assert polygon_api_key(key_file=key_file) == "test-from-file"


def test_polygon_api_key_env_wins_over_config_file(tmp_path):
    key_file = tmp_path / "polygon-massive.txt"
    key_file.write_text("default_key=from-file\n")

    with patch.dict("os.environ", {"POLYGON_API_KEY": "from-env", "BUILD_WORKING_DIRECTORY": str(tmp_path)}):
        assert polygon_api_key(key_file=key_file) == "from-env"


def test_fetch_quotes_uses_polygon_when_the_snapshot_is_today():
    snapshot = [
        {
            "ticker": "SIG",
            "updated": _ns(2026, 9, 10, 9, 31),
            "day": {"o": 103.47, "c": 96.07},
            "lastTrade": {"p": 96.07, "t": _ns(2026, 9, 10, 9, 31)},
            "prevDay": {"c": 102.48},
        }
    ]

    with (
        patch("stock_picker.ingestion.polygon_client.polygon_api_key", return_value="test-key"),
        patch("stock_picker.ingestion.polygon_client.fetch_polygon_snapshot", return_value=snapshot),
        patch("stock_picker.ingestion.yfinance_client.fetch_quote_snapshots") as mock_yahoo,
        patch("stock_picker.ingestion.finnhub_client.fetch_finnhub_quotes") as mock_finnhub,
    ):
        quotes = fetch_quotes(["SIG"], as_of=date(2026, 9, 10))

    assert quotes["SIG"]["open"] == 103.47
    mock_yahoo.assert_not_called()
    mock_finnhub.assert_not_called()


def test_fetch_quotes_falls_through_to_yahoo_when_polygon_is_stale():
    stale = [
        {
            "ticker": "SIG",
            "updated": _ns(2026, 9, 8, 16, 0),
            "day": {"o": 84.33, "c": 81.28},
            "lastTrade": {"p": 81.28, "t": _ns(2026, 9, 8, 16, 0)},
            "prevDay": {"c": 85.30},
        }
    ]
    yahoo = [
        {
            "symbol": "SIG",
            "regularMarketOpen": 103.47,
            "regularMarketPrice": 96.07,
            "regularMarketPreviousClose": 102.48,
            "regularMarketTime": int(
                datetime(2026, 9, 10, 9, 31, tzinfo=ZoneInfo("America/New_York")).timestamp()
            ),
        }
    ]

    with (
        patch("stock_picker.ingestion.polygon_client.polygon_api_key", return_value="test-key"),
        patch("stock_picker.ingestion.polygon_client.fetch_polygon_snapshot", return_value=stale),
        patch(
            "stock_picker.ingestion.yfinance_client.fetch_quote_snapshots",
            return_value=yahoo,
        ),
        patch("stock_picker.ingestion.yfinance_client.download_price_history") as mock_download,
        patch("stock_picker.ingestion.finnhub_client.fetch_finnhub_quotes", return_value={}),
    ):
        quotes = fetch_quotes(["SIG"], as_of=date(2026, 9, 10))

    assert quotes["SIG"]["open"] == 103.47
    mock_download.assert_not_called()
