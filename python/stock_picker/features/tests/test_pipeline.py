import numpy as np

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
        "seq3_open3_seasonality",
        "news_article_count_1d",
        "news_has_insider_sell_3d",
    ]:
        assert column in features.columns

    # cross-sectional columns are omitted when no benchmark/peer data is supplied,
    # same for pooled_setup_seasonality when no universe-wide input is supplied
    assert "beta_60d" not in features.columns
    assert "return_rank_5d" not in features.columns
    assert "pooled_setup_seasonality" not in features.columns


def test_zero_open_does_not_create_infinite_features():
    history = synthetic_history(n=140)
    history.loc[history.index[100], "Open"] = 0.0

    features = build_features(history)

    assert np.isnan(features.iloc[100]["body_pct"])
    assert not np.isinf(features.select_dtypes(include="number").to_numpy()).any()


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
        assert "pooled_setup_seasonality" in features.columns


def test_breadth_is_persisted_only_when_membership_history_is_supplied():
    histories = {"AAA": synthetic_history(n=80), "BBB": synthetic_history(n=80)}
    first_seen = {name: str(history.index[0].date()) for name, history in histories.items()}

    plain = build_features_for_universe(histories)
    guarded = build_features_for_universe(histories, first_seen_by_ticker=first_seen)

    assert "market_advance_fraction" not in plain["AAA"].columns
    assert "market_advance_fraction" in guarded["AAA"].columns
    assert guarded["AAA"]["market_advance_fraction"].isna().all()


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
