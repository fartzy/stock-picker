import pandas as pd
import pytest

from stock_picker.features.candle import build_candle_features
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_candle_features_matches_hand_computed_values():
    history = pd.DataFrame(
        {
            "Open": [9.0, 10.0],
            "High": [9.5, 15.0],
            "Low": [8.5, 8.0],
            "Close": [9.0, 12.0],
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


def test_build_candle_features_has_expected_columns():
    history = synthetic_history(n=10)
    features = build_candle_features(history)

    assert {
        "overnight_gap",
        "day_range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "close_location",
    } <= set(features.columns)
