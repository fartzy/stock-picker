import pandas as pd
import pytest

from stock_picker.features.pattern_seasonality import (
    build_pattern_features,
    day_session_return,
    day_session_streak,
    rolling_seasonality,
    sequence_bucket,
    weekday_lag_bucket,
    weekday_lag_return,
)
from stock_picker.features.tests.fixtures import synthetic_history


def test_day_session_return_matches_open_close_formula():
    history = pd.DataFrame({"Open": [100.0, 100.0], "Close": [110.0, 90.0]})

    returns = day_session_return(history)

    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_day_session_streak_matches_hand_computed_sequence_and_caps():
    returns = pd.Series([0.01, 0.02, -0.01, -0.02, -0.03, -0.01, 0.01])

    streak = day_session_streak(returns, cap=3)

    assert list(streak) == [1.0, 2.0, -1.0, -2.0, -3.0, -3.0, 1.0]


def test_day_session_streak_resets_to_zero_on_nan():
    returns = pd.Series([0.01, float("nan"), -0.01])

    streak = day_session_streak(returns)

    assert list(streak) == [1.0, 0.0, -1.0]


def test_sequence_bucket_is_nan_until_window_full_then_builds_key():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])

    bucket = sequence_bucket(returns, window=3)

    assert bucket.iloc[:2].isna().all()
    assert bucket.iloc[2] == "UDU"
    assert bucket.iloc[3] == "DUD"
    assert bucket.iloc[4] == "UDU"


def test_sequence_bucket_stays_nan_while_a_nan_is_inside_the_window():
    returns = pd.Series([0.01, float("nan"), 0.02, -0.02])

    bucket = sequence_bucket(returns, window=3)

    assert bucket.isna().all()


def test_weekday_lag_return_is_shifted_by_lag():
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])

    lagged = weekday_lag_return(returns, lag=5)

    pd.testing.assert_series_equal(lagged, returns.shift(5), check_names=False)


def test_weekday_lag_bucket_combines_day_of_week_and_direction_and_propagates_nan():
    dates = pd.bdate_range("2026-01-05", periods=3)
    history = pd.DataFrame({"Open": [1.0, 1.0, 1.0], "Close": [1.0, 1.0, 1.0]}, index=dates)
    lag_return = pd.Series([0.01, float("nan"), -0.01], index=dates)

    bucket = weekday_lag_bucket(history, lag_return)

    weekday = dates.dayofweek
    assert bucket.iloc[0] == f"{weekday[0]}_1.0"
    assert pd.isna(bucket.iloc[1])
    assert bucket.iloc[2] == f"{weekday[2]}_-1.0"


def test_rolling_seasonality_ages_out_old_occurrences_and_isolates_buckets():
    returns = pd.Series([1.0, 10.0, 2.0, 20.0, 3.0])
    bucket = pd.Series(["a", "b", "a", "b", "a"])

    seasonality = rolling_seasonality(returns, bucket, window=2)

    # bucket "a" (indices 0,2,4 -> values 1,2,3): rolling(2) means 1, 1.5, 2.5 --
    # index 4's value must NOT include index 0's 1.0 once the window has 2 more
    # recent "a" occurrences, which is what distinguishes this from an expanding
    # since-inception average (whose index-4 value would be (1+2+3)/3 = 2.0).
    assert list(seasonality) == [
        pytest.approx(1.0),
        pytest.approx(10.0),
        pytest.approx(1.5),
        pytest.approx(15.0),
        pytest.approx(2.5),
    ]


def test_build_pattern_features_has_expected_columns():
    history = synthetic_history(n=140)

    features = build_pattern_features(history)

    assert list(features.columns) == [
        "day_session_streak",
        "day_session_streak_seasonality",
        "pattern_sequence_seasonality_3d",
        "weekday_lag_return",
        "weekday_lag_seasonality",
    ]
    assert features["day_session_streak"].notna().all()
    assert features["weekday_lag_return"].iloc[5:].notna().any()
