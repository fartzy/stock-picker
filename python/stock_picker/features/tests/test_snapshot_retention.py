from datetime import date

import pandas as pd

from stock_picker.features.main import feature_snapshot_window


def test_retention_uses_full_prior_history_without_expanding_training_span():
    dates = pd.bdate_range("2024-01-01", "2026-01-05")
    prices = pd.Series(range(len(dates)), index=dates, dtype=float)
    features = pd.DataFrame({"long_window": prices.rolling(252).mean(), "short_window": prices})

    retained = feature_snapshot_window(features, date(2026, 1, 2))

    assert retained.index.min() > pd.Timestamp("2025-01-02")
    assert retained.index.max() == pd.Timestamp("2026-01-02")
    assert retained.iloc[0].long_window == features.loc[retained.index[0], "long_window"]
    assert retained.long_window.notna().all()
    assert list(retained.columns) == list(features.columns)
    assert len(features) > len(retained)


def test_calendar_year_boundary_handles_leap_year_and_timezone():
    dates = pd.DatetimeIndex(["2023-02-28", "2023-03-01", "2024-02-29", "2024-03-01"], tz="America/New_York")
    features = pd.DataFrame({"feature": [1, 2, 3, 4]}, index=dates)

    retained = feature_snapshot_window(features, date(2024, 2, 29))

    assert list(retained.feature) == [2, 3]


def test_empty_snapshot_keeps_schema():
    features = pd.DataFrame({"feature": []}, index=pd.DatetimeIndex([]))
    pd.testing.assert_frame_equal(feature_snapshot_window(features, date(2026, 1, 2)), features)
