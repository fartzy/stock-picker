import pandas as pd
import pytest

from stock_picker.features.open_pattern_seasonality import (
    OPEN_KNOWN_COLUMNS,
    bucket5_gap,
    build_open_pattern_features,
    completed_sequence,
    open_in_yday_range,
    open_known_feature_row,
    prior_bucket_mean,
)
from stock_picker.features.pattern_seasonality import day_session_return
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_open_pattern_features_has_the_named_open_known_columns():
    features = build_open_pattern_features(synthetic_history(n=140))

    assert list(features.columns) == OPEN_KNOWN_COLUMNS
    assert len(OPEN_KNOWN_COLUMNS) == 28
    for column in (
        "gap_seq3_open3_seasonality",
        "gap_trap_open_seasonality",
        "streak_crash_open_seasonality",
        "multi_crash_bounce_open_seasonality",
    ):
        assert column in features.columns


def test_prior_bucket_mean_excludes_the_current_row():
    values = pd.Series([1.0, 3.0, 5.0])
    bucket = pd.Series(["a", "a", "a"])

    result = prior_bucket_mean(values, bucket)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(2.0)


def test_completed_sequence_uses_prior_days_not_today():
    session = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])

    seq3 = completed_sequence(session, 3)

    assert seq3.iloc[:3].isna().all()
    assert seq3.iloc[3] == "UDU"
    assert seq3.iloc[4] == "DUD"


def test_bucket5_gap_splits_way_down_and_way_up():
    gap = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02, float("nan")])

    labels = bucket5_gap(gap)

    assert list(labels.iloc[:5]) == ["way_down", "down", "flat", "up", "way_up"]
    assert pd.isna(labels.iloc[5])


def test_open_in_yday_range_labels_below_low_mid_high_above():
    dates = pd.bdate_range("2026-01-02", periods=6)
    history = pd.DataFrame(
        {
            "Open": [10.0, 9.0, 10.2, 10.5, 10.8, 12.0],
            "High": [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            "Low": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "Close": [10.5, 10.5, 10.5, 10.5, 10.5, 10.5],
        },
        index=dates,
    )

    labels = open_in_yday_range(history)

    assert pd.isna(labels.iloc[0])
    assert labels.iloc[1] == "below"
    assert labels.iloc[2] == "low"
    assert labels.iloc[3] == "mid"
    assert labels.iloc[4] == "high"
    assert labels.iloc[5] == "above"


def test_seq2_open3_matches_hand_computed_prior_mean_of_matching_setups():
    # Two mornings share the same completed 2-day path (U then D) and a
    # down-gap open; the later morning's feature must be the earlier
    # morning's day-session return, not its own.
    dates = pd.bdate_range("2026-01-02", periods=6)
    history = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 99.0, 100.0, 101.0, 99.0],
            "High": [102.0, 102.0, 100.0, 102.0, 102.0, 100.0],
            "Low": [99.0, 99.0, 97.0, 99.0, 99.0, 97.0],
            "Close": [101.0, 99.5, 98.0, 101.0, 99.5, 100.0],
            "Volume": [1_000_000] * 6,
        },
        index=dates,
    )
    session = day_session_return(history)
    features = build_open_pattern_features(history)

    assert pd.isna(features["seq2_open3_seasonality"].iloc[4])
    assert features["seq2_open3_seasonality"].iloc[5] == pytest.approx(session.iloc[2])


def test_open_known_feature_row_matches_building_features_on_a_dummy_today_row():
    history = synthetic_history(n=80)
    today_open = float(history["Close"].iloc[-1]) * 0.99
    live = open_known_feature_row(history, today_open)

    dummy_index = history.index[-1] + pd.Timedelta(days=1)
    dummy = pd.DataFrame(
        {
            "Open": [today_open],
            "High": [today_open],
            "Low": [today_open],
            "Close": [today_open],
            "Adj Close": [today_open],
            "Volume": [0.0],
        },
        index=pd.DatetimeIndex([dummy_index]),
    )
    expected = build_open_pattern_features(pd.concat([history, dummy])).iloc[-1]
    pd.testing.assert_series_equal(live, expected, check_names=False)


def test_gap_seq3_uses_today_open_not_today_close():
    history = synthetic_history(n=80)
    yesterday_close = float(history["Close"].iloc[-1])
    down = open_known_feature_row(history, yesterday_close * 0.98)
    up = open_known_feature_row(history, yesterday_close * 1.02)

    assert down["gap_seq3_open3_seasonality"] != up["gap_seq3_open3_seasonality"] or (
        pd.isna(down["gap_seq3_open3_seasonality"]) != pd.isna(up["gap_seq3_open3_seasonality"])
    )
