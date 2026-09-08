from unittest.mock import MagicMock, patch

from stock_picker.tickers.nasdaq_directory import (
    fetch_candidate_tickers,
    fetch_nasdaq_listed_symbols,
    fetch_other_listed_symbols,
)

_FAKE_NASDAQ_LISTED = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
ZBZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N
ABCW|ABC Corp Warrants|Q|N|N|100|N|N
BRK.B|Berkshire Hathaway Inc. - Preferred Stock|Q|N|N|100|N|N
File Creation Time: 0908202610:01|||||||
"""

_FAKE_OTHER_LISTED = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
XYZ|XYZ Manufacturing Common Stock|N|XYZ|N|100|N|XYZ
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
DEFU|DEF Acquisition Corp Units|N|DEFU|N|100|N|DEFU
File Creation Time: 0908202610:01||||||
"""


def _mock_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.raise_for_status.return_value = None
    return mock_response


def test_fetch_nasdaq_listed_symbols_filters_etfs_test_issues_warrants_and_preferred():
    with patch(
        "stock_picker.tickers.nasdaq_directory.requests.get",
        return_value=_mock_response(_FAKE_NASDAQ_LISTED),
    ):
        tickers = fetch_nasdaq_listed_symbols()

    assert tickers == ["AAPL"]


def test_fetch_other_listed_symbols_filters_etfs_and_spac_units():
    with patch(
        "stock_picker.tickers.nasdaq_directory.requests.get",
        return_value=_mock_response(_FAKE_OTHER_LISTED),
    ):
        tickers = fetch_other_listed_symbols()

    assert tickers == ["XYZ"]


def test_fetch_candidate_tickers_combines_and_dedupes_both_directories():
    with patch(
        "stock_picker.tickers.nasdaq_directory.requests.get",
        side_effect=[_mock_response(_FAKE_NASDAQ_LISTED), _mock_response(_FAKE_OTHER_LISTED)],
    ):
        tickers = fetch_candidate_tickers()

    assert tickers == ["AAPL", "XYZ"]
