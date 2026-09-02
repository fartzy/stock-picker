import pandas as pd
import pytest

from stock_picker.features.distributional import (
    WINDOWS,
    build_distributional_features,
    return_autocorrelation,
    rolling_drawdown,
)
from stock_picker.features.tests.fixtures import synthetic_history


def test_rolling_drawdown_matches_hand_computed_values():
    close = pd.Series([10.0, 12.0, 8.0, 9.0])

    drawdown = rolling_drawdown(close, window=3)

    assert drawdown.iloc[0] == pytest.approx(0.0)
    assert drawdown.iloc[1] == pytest.approx(0.0)
    assert drawdown.iloc[2] == pytest.approx(8.0 / 12.0 - 1)
    assert drawdown.iloc[3] == pytest.approx(9.0 / 12.0 - 1)


def test_return_autocorrelation_is_1_for_a_linear_series():
    returns = pd.Series([float(i) for i in range(10)])

    result = return_autocorrelation(returns, window=5, lag=1)

    assert result.iloc[-1] == pytest.approx(1.0)


def test_build_distributional_features_has_expected_columns():
    history = synthetic_history(n=80)
    features = build_distributional_features(history)

    for n in WINDOWS:
        for prefix in ["skew", "kurtosis", "max_drawdown", "return_autocorr", "sharpe", "var_5pct"]:
            assert f"{prefix}_{n}d" in features.columns
