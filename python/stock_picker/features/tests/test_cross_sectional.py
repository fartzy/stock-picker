import pandas as pd
import pytest

from stock_picker.features.cross_sectional import (
    build_cross_sectional_features,
    relative_strength,
    return_rank,
    rolling_beta,
)
from stock_picker.features.tests.fixtures import synthetic_history


def test_return_rank_matches_hand_computed_percentiles():
    wide = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 1.0], "C": [3.0, 2.0]})

    ranks = return_rank(wide)

    assert ranks.loc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert ranks.loc[1].tolist() == pytest.approx([1.0, 1 / 3, 2 / 3])


def test_rolling_beta_is_2_when_ticker_moves_twice_the_benchmark():
    benchmark_returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.015, -0.005])
    ticker_returns = benchmark_returns * 2

    beta = rolling_beta(ticker_returns, benchmark_returns, window=5)

    assert beta.iloc[-1] == pytest.approx(2.0)


def test_relative_strength_matches_hand_computed_value():
    ticker_close = pd.Series([100.0, 200.0])
    benchmark_close = pd.Series([50.0, 50.0])

    result = relative_strength(ticker_close, benchmark_close)

    assert result.iloc[-1] == pytest.approx(1.0)


def test_build_cross_sectional_features_omits_columns_for_missing_inputs():
    history = synthetic_history(n=10)

    features = build_cross_sectional_features(history)

    assert features.empty or len(features.columns) == 0


def test_build_cross_sectional_features_includes_benchmark_columns_when_provided():
    history = synthetic_history(n=80)
    benchmark_history = synthetic_history(n=80)

    features = build_cross_sectional_features(history, benchmark_history=benchmark_history)

    assert {"beta_60d", "correlation_60d", "relative_strength"} <= set(features.columns)
