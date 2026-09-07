import pandas as pd
import pytest

from stock_picker.features.candle import build_candle_features, gap_streak
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_candle_features_matches_hand_computed_values():
    history = pd.DataFrame(
        {
            "Open": [9.0, 10.0],
            "High": [9.5, 15.0],
            "Low": [8.5, 8.0],
            "Close": [9.0, 12.0],
            "Volume": [1_000_000, 1_100_000],
        }
    )

    features = build_candle_features(history)
    row = features.iloc[1]

    assert row["overnight_gap"] == pytest.approx((10.0 - 9.0) / 9.0)
    assert row["day_range_pct"] == pytest.approx((15.0 - 8.0) / 12.0)
    assert row["body_pct"] == pytest.approx((12.0 - 10.0) / 10.0)
    assert row["upper_wick_pct"] == pytest.approx((15.0 - 12.0) / 10.0)
    assert row["lower_wick_pct"] == pytest.approx((10.0 - 8.0) / 10.0)
    assert row["close_location"] == pytest.approx((12.0 - 8.0) / (15.0 - 8.0))
    # Only 2 rows -- volume_zscore's 20-day rolling window hasn't warmed up,
    # so the interaction is correctly NaN, not a fabricated number.
    assert pd.isna(row["gap_volume_interaction"])
    assert row["gap_streak"] == pytest.approx(1.0)  # the only gap so far is up


def test_gap_streak_counts_consecutive_same_direction_gaps():
    gap = pd.Series([0.01, 0.02, -0.01, -0.02, -0.03, 0.0, 0.01])

    streak = gap_streak(gap)

    assert list(streak) == [1.0, 2.0, -1.0, -2.0, -3.0, 0.0, 1.0]


def test_gap_volume_interaction_has_real_values_once_the_volume_window_warms_up():
    history = synthetic_history(n=30)

    features = build_candle_features(history)

    assert features["gap_volume_interaction"].notna().any()


def test_build_candle_features_has_expected_columns():
    history = synthetic_history(n=10)
    features = build_candle_features(history)

    assert {
        "overnight_gap",
        "gap_volume_interaction",
        "gap_streak",
        "day_range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "close_location",
    } <= set(features.columns)
