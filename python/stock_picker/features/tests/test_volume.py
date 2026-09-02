import pandas as pd
import pytest

from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.features.volume import (
    VOLUME_RATIO_WINDOWS,
    build_volume_features,
    on_balance_volume,
    volume_ratio,
)


def test_on_balance_volume_matches_hand_computed_values():
    close = pd.Series([10.0, 11.0, 9.0, 9.0, 12.0])
    volume = pd.Series([100, 200, 300, 400, 500])

    obv = on_balance_volume(close, volume)

    assert list(obv) == [0, 200, -100, -100, 400]


def test_volume_ratio_matches_hand_computed_value():
    volume = pd.Series([10, 20, 30])

    result = volume_ratio(volume, window=3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(1.5)


def test_build_volume_features_has_expected_columns():
    history = synthetic_history(n=80)
    features = build_volume_features(history)

    for n in VOLUME_RATIO_WINDOWS:
        assert f"volume_ratio_{n}d" in features.columns
    assert {"dollar_volume", "obv", "obv_change_20d", "volume_zscore_20d"} <= set(
        features.columns
    )
