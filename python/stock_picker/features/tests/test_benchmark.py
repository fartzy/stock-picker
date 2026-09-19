from unittest.mock import patch

import pandas as pd
import pytest

from stock_picker.features.benchmark import (
    fetch_benchmark_hold,
    fetch_benchmark_overnight,
    fetch_benchmark_returns,
)


def _fake_spy_history() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [500.0, 510.0], "Close": [505.0, 504.9]},
        index=pd.to_datetime(["2026-09-04", "2026-09-08"]),
    )


def test_fetch_benchmark_returns_computes_day_session_return():
    with patch(
        "stock_picker.features.benchmark.download_price_history",
        return_value={"SPY": _fake_spy_history()},
    ):
        returns = fetch_benchmark_returns(["2026-09-04", "2026-09-08"])

    assert returns["2026-09-04"] == (505.0 - 500.0) / 500.0
    assert returns["2026-09-08"] == (504.9 - 510.0) / 510.0


def test_fetch_benchmark_returns_omits_dates_with_no_trading_data():
    with patch(
        "stock_picker.features.benchmark.download_price_history",
        return_value={"SPY": _fake_spy_history()},
    ):
        returns = fetch_benchmark_returns(["2026-09-04", "2026-09-05"])  # 9/5 is a Saturday

    assert set(returns.keys()) == {"2026-09-04"}


def test_fetch_benchmark_returns_empty_input_skips_the_fetch():
    with patch("stock_picker.features.benchmark.download_price_history") as mock_download:
        returns = fetch_benchmark_returns([])

    mock_download.assert_not_called()
    assert returns == {}


def test_fetch_benchmark_overnight_is_prior_close_to_this_close():
    with patch(
        "stock_picker.features.benchmark.download_price_history",
        return_value={"SPY": _fake_spy_history()},
    ):
        overnight = fetch_benchmark_overnight(["2026-09-04", "2026-09-08"])

    assert "2026-09-04" not in overnight  # no prior close in the window
    assert overnight["2026-09-08"] == pytest.approx((504.9 / 505.0) - 1.0)


def test_fetch_benchmark_hold_is_close_to_close_not_session_average():
    with patch(
        "stock_picker.features.benchmark.download_price_history",
        return_value={"SPY": _fake_spy_history()},
    ):
        hold = fetch_benchmark_hold("2026-09-04", "2026-09-08")

    # 504.9 / 505 - 1, not the average of the two open->close days.
    assert hold["from"] == "2026-09-04"
    assert hold["to"] == "2026-09-08"
    assert hold["return"] == pytest.approx((504.9 / 505.0) - 1.0)
