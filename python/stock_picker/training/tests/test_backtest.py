import pandas as pd
import pytest

from stock_picker.training.backtest import simulate_trades, sweep_thresholds


def test_simulate_trades_hit_rate_and_total_return_above_threshold():
    predicted = pd.Series([0.03, 0.02, 0.01, -0.01, 0.0])
    actual = pd.Series([0.05, 0.04, -0.02, 0.01, -0.01])

    result = simulate_trades(predicted, actual, threshold=0.015)

    # only the first two rows (predicted 0.03, 0.02) clear the threshold
    assert result["n_trades"] == 2
    assert result["hit_rate"] == 1.0
    assert result["total_return"] == pytest.approx(0.09)
    assert result["avg_return"] == pytest.approx(0.045)


def test_simulate_trades_lower_threshold_includes_a_losing_trade():
    predicted = pd.Series([0.03, 0.02, 0.01, -0.01, 0.0])
    actual = pd.Series([0.05, 0.04, -0.02, 0.01, -0.01])

    result = simulate_trades(predicted, actual, threshold=0.005)

    # rows with predicted 0.03, 0.02, 0.01 clear the threshold; the third is a loser
    assert result["n_trades"] == 3
    assert result["hit_rate"] == pytest.approx(2 / 3)


def test_simulate_trades_no_trades_returns_nan_not_error():
    predicted = pd.Series([0.0, -0.01, -0.02])
    actual = pd.Series([0.01, 0.02, 0.03])

    result = simulate_trades(predicted, actual, threshold=0.5)

    assert result["n_trades"] == 0
    assert pd.isna(result["hit_rate"])
    assert pd.isna(result["total_return"])


def test_sweep_thresholds_returns_one_row_per_threshold():
    predicted = pd.Series([0.03, 0.02, 0.01, -0.01, 0.0])
    actual = pd.Series([0.05, 0.04, -0.02, 0.01, -0.01])

    sweep = sweep_thresholds(predicted, actual, thresholds=[0.0, 0.01, 0.02])

    assert len(sweep) == 3
    assert list(sweep["threshold"]) == [0.0, 0.01, 0.02]


def test_simulate_trades_without_n_days_omits_avg_picks_per_day():
    predicted = pd.Series([0.03, 0.02, 0.01, -0.01, 0.0])
    actual = pd.Series([0.05, 0.04, -0.02, 0.01, -0.01])

    result = simulate_trades(predicted, actual, threshold=0.015)

    assert result["avg_picks_per_day"] is None


def test_simulate_trades_with_n_days_reports_average_picks_per_day():
    # 2 trades clear the threshold (see the hit-rate test above), spread
    # across a stand-in 5-day evaluation window -- answers "how many stocks
    # would this flag on a given day," not just "how many trades total."
    predicted = pd.Series([0.03, 0.02, 0.01, -0.01, 0.0])
    actual = pd.Series([0.05, 0.04, -0.02, 0.01, -0.01])

    result = simulate_trades(predicted, actual, threshold=0.015, n_days=5)

    assert result["n_trades"] == 2
    assert result["avg_picks_per_day"] == pytest.approx(0.4)
