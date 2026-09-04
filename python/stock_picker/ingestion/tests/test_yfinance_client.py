from unittest.mock import patch

import pandas as pd

from stock_picker.ingestion.yfinance_client import download_price_history


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
