import pandas as pd

from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.features.volatility import (
    RANGE_VOL_WINDOWS,
    REALIZED_VOL_WINDOWS,
    average_true_range,
    build_volatility_features,
)


def test_average_true_range_matches_hand_computed_values():
    history = pd.DataFrame(
        {
            "Open": [9.0, 10.0, 8.0],
            "High": [10.0, 11.0, 9.0],
            "Low": [8.0, 9.0, 6.0],
            "Close": [9.0, 10.0, 7.0],
        }
    )

    atr = average_true_range(history, window=2)

    # TR: row0=2 (no prior close), row1=max(2,2,0)=2, row2=max(3,1,4)=4
    assert pd.isna(atr.iloc[0])
    assert atr.iloc[1] == 2.0
    assert atr.iloc[2] == 3.0


def test_build_volatility_features_has_expected_columns():
    history = synthetic_history(n=140)
    features = build_volatility_features(history)

    for n in REALIZED_VOL_WINDOWS:
        assert f"volatility_{n}d" in features.columns
    assert "atr_14" in features.columns
    for n in RANGE_VOL_WINDOWS:
        assert f"parkinson_vol_{n}d" in features.columns
        assert f"garman_klass_vol_{n}d" in features.columns

    # range-based estimators should be non-negative wherever defined
    for n in RANGE_VOL_WINDOWS:
        col = features[f"parkinson_vol_{n}d"].dropna()
        assert (col >= 0).all()
