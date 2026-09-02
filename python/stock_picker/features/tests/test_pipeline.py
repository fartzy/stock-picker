from stock_picker.features.pipeline import build_features, build_features_for_universe
from stock_picker.features.tests.fixtures import synthetic_history


def test_build_features_combines_every_category():
    history = synthetic_history(n=140)

    features = build_features(history)

    for column in [
        "return_1d",
        "volatility_20d",
        "price_vs_sma_20d",
        "rsi_14",
        "volume_ratio_20d",
        "overnight_gap",
        "skew_20d",
        "day_of_week",
    ]:
        assert column in features.columns

    # cross-sectional columns are omitted when no benchmark/peer data is supplied
    assert "beta_60d" not in features.columns
    assert "return_rank_5d" not in features.columns


def test_build_features_for_universe_adds_return_rank_and_beta_columns():
    histories = {
        "AAA": synthetic_history(n=80),
        "BBB": synthetic_history(n=80),
    }
    benchmark_history = synthetic_history(n=80)

    features_by_ticker = build_features_for_universe(
        histories, benchmark_history=benchmark_history
    )

    assert set(features_by_ticker) == {"AAA", "BBB"}
    for features in features_by_ticker.values():
        assert "return_rank_5d" in features.columns
        assert "beta_60d" in features.columns


def test_build_features_for_universe_ranks_are_valid_percentiles():
    histories = {
        "AAA": synthetic_history(n=40),
        "BBB": synthetic_history(n=40),
        "CCC": synthetic_history(n=40),
    }

    features_by_ticker = build_features_for_universe(histories)

    for features in features_by_ticker.values():
        ranks = features["return_rank_5d"].dropna()
        assert ((ranks > 0) & (ranks <= 1)).all()
