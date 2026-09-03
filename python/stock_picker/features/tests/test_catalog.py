import pandas as pd
import pytest

from stock_picker.features.catalog import coverage_report, list_feature_columns
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
        "cross_sectional",
    }
    assert set(catalog) == expected_categories
    for columns in catalog.values():
        assert len(columns) > 0

    total_columns = sum(len(columns) for columns in catalog.values())
    assert total_columns >= 80  # 79 before this pass + 4 new, some slack


def test_coverage_report_flags_an_all_nan_column():
    table_a = pd.DataFrame({"good": [1.0, 2.0, 3.0], "bad": [None, None, None]})
    table_b = pd.DataFrame({"good": [1.0, None, 3.0], "bad": [None, None, None]})

    report = coverage_report({"A": table_a, "B": table_b})

    assert report.loc["bad", "non_null_pct"] == 0.0
    assert report.loc["good", "non_null_pct"] == pytest.approx((1.0 + 2 / 3) / 2)
    # sorted ascending -- the all-NaN column should be first
    assert report.index[0] == "bad"
