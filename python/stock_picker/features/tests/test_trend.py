import pandas as pd
import pytest

from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.features.trend import (
    EMA_WINDOWS,
    SMA_WINDOWS,
    build_trend_features,
    ema,
    price_vs_sma,
)


def test_ema_matches_hand_computed_values():
    close = pd.Series([10.0, 12.0, 11.0])

    result = ema(close, span=2)

    # alpha = 2/(2+1) = 2/3; seed = first value
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(11.333333, rel=1e-5)
    assert result.iloc[2] == pytest.approx(11.111111, rel=1e-5)


def test_price_vs_sma_matches_hand_computed_value():
    close = pd.Series([10.0, 20.0, 30.0])

    result = price_vs_sma(close, window=3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(0.5)


def test_build_trend_features_has_expected_columns():
    history = synthetic_history(n=140)
    features = build_trend_features(history)

    for n in SMA_WINDOWS:
        assert f"price_vs_sma_{n}d" in features.columns
    for n in EMA_WINDOWS:
        assert f"price_vs_ema_{n}d" in features.columns
    assert {"macd_line", "macd_signal", "macd_hist"} <= set(features.columns)
