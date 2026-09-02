import pandas as pd
import pytest

from stock_picker.features.oscillators import (
    RSI_WINDOWS,
    bollinger_bands,
    build_oscillator_features,
    rsi,
    stochastic_oscillator,
    williams_r,
)
from stock_picker.features.tests.fixtures import synthetic_history


def test_rsi_is_100_for_pure_uptrend():
    close = pd.Series([10.0 + i for i in range(15)])
    assert rsi(close, window=3).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_for_pure_downtrend():
    close = pd.Series([30.0 - i for i in range(15)])
    assert rsi(close, window=3).iloc[-1] == pytest.approx(0.0)


def test_stochastic_and_williams_r_at_period_high():
    history = pd.DataFrame(
        {
            "High": [10.0, 11.0, 12.0, 13.0, 14.0],
            "Low": [8.0, 9.0, 10.0, 11.0, 12.0],
            "Close": [9.0, 10.0, 11.0, 12.0, 14.0],
        }
    )

    percent_k, _ = stochastic_oscillator(history, window=5, signal_window=1)
    r = williams_r(history, window=5)

    assert percent_k.iloc[-1] == pytest.approx(100.0)
    assert r.iloc[-1] == pytest.approx(0.0)


def test_bollinger_bands_matches_direct_recomputation():
    close = pd.Series([100.0 + (i % 5) for i in range(30)])

    percent_b, bandwidth = bollinger_bands(close, window=10, num_std=2)

    mid = close.rolling(10).mean()
    std = close.rolling(10).std()
    expected_upper = mid + 2 * std
    expected_lower = mid - 2 * std
    expected_percent_b = (close - expected_lower) / (expected_upper - expected_lower)
    expected_bandwidth = (expected_upper - expected_lower) / mid

    pd.testing.assert_series_equal(percent_b, expected_percent_b, check_names=False)
    pd.testing.assert_series_equal(bandwidth, expected_bandwidth, check_names=False)


def test_build_oscillator_features_has_expected_columns():
    history = synthetic_history(n=60)
    features = build_oscillator_features(history)

    for n in RSI_WINDOWS:
        assert f"rsi_{n}" in features.columns
    assert {
        "stochastic_k",
        "stochastic_d",
        "bollinger_pct_b",
        "bollinger_bandwidth",
        "williams_r",
        "cci",
    } <= set(features.columns)
