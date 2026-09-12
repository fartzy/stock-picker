import pandas as pd
import pytest

from stock_picker.features.pipeline import build_features_for_universe
from stock_picker.features.regime import build_regime_features, spy_overnight_gap
from stock_picker.features.tests.fixtures import synthetic_history


def test_spy_overnight_gap_uses_open_vs_prior_close():
    spy = pd.DataFrame(
        {"Open": [101.0, 102.0], "Close": [100.0, 103.0]},
        index=pd.bdate_range("2026-01-02", periods=2),
    )

    gap = spy_overnight_gap(spy)

    assert pd.isna(gap.iloc[0])
    assert gap.iloc[1] == pytest.approx(0.02)


def test_build_regime_features_broadcasts_onto_the_ticker_index():
    history = synthetic_history(n=10)
    spy = synthetic_history(n=10)
    vix = synthetic_history(n=10)

    features = build_regime_features(history.index, spy_history=spy, vix_history=vix)

    assert list(features.columns) == ["spy_overnight_gap", "vix_close"]
    assert len(features) == 10


def test_universe_pipeline_fills_sector_relative_return_when_labels_exist():
    histories = {
        "AAA": synthetic_history(n=40),
        "BBB": synthetic_history(n=40),
    }

    features = build_features_for_universe(
        histories, sector_by_ticker={"AAA": "Tech", "BBB": "Tech"}
    )

    assert "sector_relative_return" in features["AAA"].columns
    assert features["AAA"]["sector_relative_return"].notna().any()
