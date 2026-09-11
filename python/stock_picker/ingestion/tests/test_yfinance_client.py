from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.ingestion.yfinance_client import (
    download_price_history,
    fetch_quotes,
    quotes_from_history,
    quotes_from_intraday,
    quotes_from_snapshot,
)


def test_download_price_history_splits_by_ticker():
    columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "Close"]])
    index = pd.date_range("2026-01-01", periods=2)
    raw = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]],
        index=index,
        columns=columns,
    )

    with patch("stock_picker.ingestion.yfinance_client.yf.download", return_value=raw) as mock_download:
        result = download_price_history(["AAPL", "MSFT"], period="6mo")

    mock_download.assert_called_once()
    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert list(result["AAPL"].columns) == ["Open", "Close"]


def test_download_price_history_slices_multiindex_for_a_single_ticker():
    # yfinance still returns MultiIndex (ticker, field) columns for one ticker
    # when group_by="ticker" is set -- this must be sliced just like the
    # multi-ticker case, not returned as-is.
    columns = pd.MultiIndex.from_product([["SPY"], ["Open", "Close"]])
    index = pd.date_range("2026-01-01", periods=2)
    raw = pd.DataFrame([[1.0, 2.0], [1.1, 2.1]], index=index, columns=columns)

    with patch("stock_picker.ingestion.yfinance_client.yf.download", return_value=raw):
        result = download_price_history(["SPY"], period="6mo")

    assert set(result.keys()) == {"SPY"}
    assert list(result["SPY"].columns) == ["Open", "Close"]


def test_download_price_history_excludes_a_failed_ticker():
    # a ticker yfinance couldn't fetch (delisted, transient failure) comes back as
    # an all-NaN slice -- it must be excluded, not stored as an empty DataFrame.
    columns = pd.MultiIndex.from_product([["AAPL", "OMC"], ["Open", "Close"]])
    index = pd.date_range("2026-01-01", periods=2)
    raw = pd.DataFrame(
        [[1.0, 2.0, None, None], [1.1, 2.1, None, None]],
        index=index,
        columns=columns,
    )

    with patch("stock_picker.ingestion.yfinance_client.yf.download", return_value=raw):
        result = download_price_history(["AAPL", "OMC"], period="6mo")

    assert set(result.keys()) == {"AAPL"}


def test_download_price_history_defaults_to_one_year():
    columns = pd.MultiIndex.from_product([["AAPL"], ["Open", "Close"]])
    raw = pd.DataFrame([[1.0, 2.0]], index=pd.date_range("2026-01-01", periods=1), columns=columns)

    with patch(
        "stock_picker.ingestion.yfinance_client.yf.download", return_value=raw
    ) as mock_download:
        download_price_history(["AAPL"])

    assert mock_download.call_args.kwargs["period"] == "1y"


def _daily(rows, start="2026-09-08"):
    index = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close", "Volume"])


def test_quotes_from_history_uses_the_today_bar_not_iloc_last():
    # Yesterday 50 / today 51 -- iloc[-1] happens to be right here; the
    # as_of filter is what the next test proves is load-bearing.
    history = _daily(
        [[50.0, 51.0, 49.0, 50.5, 1000], [51.0, 52.0, 50.0, 51.5, 1100]],
        start="2026-09-08",
    )

    quotes = quotes_from_history({"MKC": history}, as_of=date(2026, 9, 9))

    assert quotes["MKC"]["open"] == 51.0
    assert quotes["MKC"]["last"] == 51.5
    assert quotes["MKC"]["prev_close"] == 50.5


def test_quotes_from_history_omits_a_ticker_with_no_today_bar():
    yesterday_only = _daily([[50.0, 51.0, 49.0, 50.5, 1000]], start="2026-09-08")

    quotes = quotes_from_history({"MKC": yesterday_only}, as_of=date(2026, 9, 9))

    assert quotes == {}


def test_quotes_from_history_keeps_a_large_today_open():
    history = _daily(
        [[52.0, 53.0, 51.0, 52.2, 5000], [95.01, 96.0, 51.0, 52.0, 200]],
        start="2026-09-08",
    )

    quotes = quotes_from_history({"SIG": history}, as_of=date(2026, 9, 9))

    assert quotes["SIG"]["open"] == 95.01


def _eastern_unix(year, month, day, hour=9, minute=30):
    dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    return int(dt.timestamp())


def test_quotes_from_snapshot_uses_regular_market_open_from_today():
    raw = [
        {
            "symbol": "SIG",
            "regularMarketOpen": 103.47,
            "regularMarketPrice": 96.07,
            "regularMarketPreviousClose": 102.48,
            "regularMarketTime": _eastern_unix(2026, 9, 10, 9, 31),
        }
    ]

    quotes = quotes_from_snapshot(raw, as_of=date(2026, 9, 10))

    assert quotes["SIG"] == {"open": 103.47, "last": 96.07, "prev_close": 102.48}


def test_quotes_from_snapshot_omits_a_stale_previous_session_quote():
    raw = [
        {
            "symbol": "SIG",
            "regularMarketOpen": 84.33,
            "regularMarketPrice": 81.28,
            "regularMarketPreviousClose": 85.30,
            "regularMarketTime": _eastern_unix(2026, 9, 8, 16, 0),
        }
    ]

    quotes = quotes_from_snapshot(raw, as_of=date(2026, 9, 10))

    assert quotes == {}


def test_quotes_from_snapshot_keeps_a_large_today_open():
    raw = [
        {
            "symbol": "TTAN",
            "regularMarketOpen": 56.35,
            "regularMarketPrice": 56.40,
            "regularMarketPreviousClose": 85.68,
            "regularMarketTime": _eastern_unix(2026, 9, 10, 9, 31),
        }
    ]

    quotes = quotes_from_snapshot(raw, as_of=date(2026, 9, 10))

    assert quotes["TTAN"]["open"] == 56.35


def test_quotes_from_intraday_uses_first_regular_session_print_on_as_of():
    index = pd.DatetimeIndex(
        [
            "2026-09-09 15:59:00",
            "2026-09-10 09:29:00",
            "2026-09-10 09:30:00",
            "2026-09-10 09:31:00",
        ],
        tz="America/New_York",
    )
    history = pd.DataFrame(
        {
            "Open": [80.0, 99.0, 103.47, 103.10],
            "High": [81.0, 99.0, 103.50, 103.20],
            "Low": [79.0, 99.0, 103.00, 102.90],
            "Close": [80.5, 99.0, 103.20, 102.80],
            "Volume": [1000, 10, 500, 400],
        },
        index=index,
    )

    quotes = quotes_from_intraday({"SIG": history}, as_of=date(2026, 9, 10))

    assert quotes["SIG"]["open"] == 103.47
    assert quotes["SIG"]["last"] == 102.80
    assert quotes["SIG"]["prev_close"] == 80.5


def test_quotes_from_intraday_omits_a_ticker_with_no_today_rth_bar():
    index = pd.DatetimeIndex(["2026-09-09 15:59:00"], tz="America/New_York")
    history = pd.DataFrame(
        {"Open": [80.0], "High": [81.0], "Low": [79.0], "Close": [80.5], "Volume": [1000]},
        index=index,
    )

    quotes = quotes_from_intraday({"SIG": history}, as_of=date(2026, 9, 10))

    assert quotes == {}


def test_fetch_quotes_falls_back_to_intraday_when_yahoo_snapshot_is_not_today():
    stale_snapshot = [
        {
            "symbol": "SIG",
            "regularMarketOpen": 84.33,
            "regularMarketPrice": 81.28,
            "regularMarketPreviousClose": 85.30,
            "regularMarketTime": _eastern_unix(2026, 9, 8, 16, 0),
        }
    ]
    index = pd.DatetimeIndex(
        ["2026-09-09 15:59:00", "2026-09-10 09:30:00"],
        tz="America/New_York",
    )
    minute_history = pd.DataFrame(
        {
            "Open": [80.0, 103.47],
            "High": [81.0, 103.50],
            "Low": [79.0, 103.00],
            "Close": [80.5, 103.20],
            "Volume": [1000, 500],
        },
        index=index,
    )

    with (
        patch("stock_picker.ingestion.polygon_client.fetch_polygon_quotes", return_value={}),
        patch("stock_picker.ingestion.finnhub_client.fetch_finnhub_quotes", return_value={}),
        patch(
            "stock_picker.ingestion.yfinance_client.fetch_quote_snapshots",
            return_value=stale_snapshot,
        ),
        patch(
            "stock_picker.ingestion.yfinance_client.download_price_history",
            return_value={"SIG": minute_history},
        ) as mock_download,
    ):
        quotes = fetch_quotes(["SIG"], as_of=date(2026, 9, 10))

    assert quotes["SIG"]["open"] == 103.47
    assert mock_download.call_args.kwargs["interval"] == "1m"
