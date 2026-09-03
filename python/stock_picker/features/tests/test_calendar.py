import pandas as pd
import pytest

from stock_picker.features.calendar import build_calendar_features, day_of_week_seasonality


def test_build_calendar_features_matches_known_dates():
    history = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-31"]),
    )

    features = build_calendar_features(history)

    # 2026-03-02 is a Monday (day_of_week=0), 2026-03-03 a Tuesday (1)
    assert features["day_of_week"].tolist() == [0, 1, 1]
    assert features["day_of_month"].tolist() == [2, 3, 31]
    assert features["month"].tolist() == [3, 3, 3]
    assert "day_of_week_seasonality" in features.columns


def test_day_of_week_seasonality_matches_hand_computed_expanding_mean():
    # dates 7 days apart always share a weekday, regardless of which weekday it is
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"])
    history = pd.DataFrame({"Close": [100.0, 102.0, 103.0, 106.0, 104.0]}, index=dates)

    seasonality = day_of_week_seasonality(history)

    first_repeat_return = (102.0 - 100.0) / 100.0
    second_repeat_return = (106.0 - 103.0) / 103.0
    expected_at_second_occurrence = (first_repeat_return + second_repeat_return) / 2

    assert seasonality.iloc[1] == pytest.approx(first_repeat_return)
    assert seasonality.iloc[3] == pytest.approx(expected_at_second_occurrence)
