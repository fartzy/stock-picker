import pandas as pd
import pytest

from stock_picker.features.catalog import (
    correlation_matrix,
    coverage_report,
    experimental_features,
    list_feature_columns,
    model_derived_features,
    top_correlated_pairs,
)
from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.features.stacked_svm import PRODUCTION_MODEL_DERIVED_COLUMNS, RESEARCH_SVM_COLUMNS, STACKED_SVM_COLUMNS


def test_list_feature_columns_has_all_categories():
    history = synthetic_history(n=140)

    catalog = list_feature_columns(history)

    expected_categories = {
        "momentum",
        "volatility",
        "trend",
        "oscillators",
        "volume",
        "candle",
        "distributional",
        "calendar",
        "conditional_seasonality",
        "cross_sectional",
        "pattern_seasonality",
        "open_pattern_seasonality",
        "regime",
        "news",
        "structure",
        "weather",
    }
    assert set(catalog) == expected_categories
    for columns in catalog.values():
        assert len(columns) > 0

    total_columns = sum(len(columns) for columns in catalog.values())
    # 95 pre-pass + setup_seasonality + pooled_setup_seasonality, some slack
    assert total_columns >= 90
    assert not set(STACKED_SVM_COLUMNS).intersection(
        column for columns in catalog.values() for column in columns
    )


def test_experimental_features_are_described_without_pipeline_formulas():
    experimental = experimental_features()

    assert tuple(experimental) == RESEARCH_SVM_COLUMNS
    for name, feature in experimental.items():
        assert feature.name == name
        assert feature.description and feature.computation and feature.example
        assert "earlier" in feature.computation.lower()


def test_production_model_derived_feature_is_separate_from_research_and_pipeline():
    derived = model_derived_features()

    assert tuple(derived) == PRODUCTION_MODEL_DERIVED_COLUMNS
    assert not set(derived) & set(experimental_features())
    assert derived["svc_direction_margin"].computation


def test_coverage_report_flags_an_all_nan_column():
    table_a = pd.DataFrame({"good": [1.0, 2.0, 3.0], "bad": [None, None, None]})
    table_b = pd.DataFrame({"good": [1.0, None, 3.0], "bad": [None, None, None]})

    report = coverage_report({"A": table_a, "B": table_b})

    assert report.loc["bad", "non_null_pct"] == 0.0
    assert report.loc["good", "non_null_pct"] == pytest.approx((1.0 + 2 / 3) / 2)
    # sorted ascending -- the all-NaN column should be first
    assert report.index[0] == "bad"


def test_top_correlated_pairs_finds_a_perfectly_correlated_pair():
    table = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0], "z": [4.0, 1.0, 3.0, 2.0]})

    corr = correlation_matrix({"AAA": table, "BBB": table})
    pairs = top_correlated_pairs(corr, n=5)

    assert pairs[0]["a"] == "x"
    assert pairs[0]["b"] == "y"
    assert pairs[0]["correlation"] == pytest.approx(1.0)
