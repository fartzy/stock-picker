import pandas as pd
import pytest

from stock_picker.features.momentum import (
    SPREAD_WINDOW_PAIRS,
    WINDOWS,
    build_momentum_features,
    consecutive_day_streak,
    momentum_spread,
)
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_momentum_features_has_expected_columns():
    history = synthetic_history(n=140)
    features = build_momentum_features(history)

    for n in WINDOWS:
        assert f"return_{n}d" in features.columns
        assert f"log_return_{n}d" in features.columns
    for short_window, long_window in SPREAD_WINDOW_PAIRS:
        assert f"momentum_spread_{short_window}_{long_window}d" in features.columns
    assert "consecutive_day_streak" in features.columns


def test_return_1d_matches_pct_change():
    history = synthetic_history(n=10)
    features = build_momentum_features(history)

    expected = history["Close"].pct_change(1)
    pd.testing.assert_series_equal(features["return_1d"], expected, check_names=False)


def test_momentum_spread_matches_hand_computed_value():
    close = pd.Series([100.0, 110.0, 90.0])

    spread = momentum_spread(close, short_window=1, long_window=2)

    return_1d = (90.0 - 110.0) / 110.0
    return_2d = (90.0 - 100.0) / 100.0
    assert spread.iloc[2] == pytest.approx(return_1d - return_2d)


def test_consecutive_day_streak_matches_hand_computed_sequence():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0])

    streak = consecutive_day_streak(close)

    assert list(streak) == [0.0, 1.0, 2.0, -1.0, -2.0, -3.0, 1.0]
