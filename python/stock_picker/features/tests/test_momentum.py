import pandas as pd

from stock_picker.features.momentum import WINDOWS, build_momentum_features
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_momentum_features_has_expected_columns():
    history = synthetic_history(n=140)
    features = build_momentum_features(history)

    for n in WINDOWS:
        assert f"return_{n}d" in features.columns
        assert f"log_return_{n}d" in features.columns


def test_return_1d_matches_pct_change():
    history = synthetic_history(n=10)
    features = build_momentum_features(history)

    expected = history["Close"].pct_change(1)
    pd.testing.assert_series_equal(features["return_1d"], expected, check_names=False)
