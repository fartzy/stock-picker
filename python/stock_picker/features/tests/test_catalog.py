import pandas as pd
import pytest

from stock_picker.features.catalog import (
    correlation_matrix,
    coverage_report,
    list_feature_columns,
    top_correlated_pairs,
)
from stock_picker.features.tests.fixtures import synthetic_history


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
    }
    assert set(catalog) == expected_categories
    for columns in catalog.values():
        assert len(columns) > 0

    total_columns = sum(len(columns) for columns in catalog.values())
    # 95 pre-pass + setup_seasonality + pooled_setup_seasonality, some slack
    assert total_columns >= 90


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
