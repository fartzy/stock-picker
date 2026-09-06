from stock_picker.features.catalog import list_feature_columns
from stock_picker.features.formulas import UNKNOWN_FORMULA, describe_computation
from stock_picker.features.tests.fixtures import synthetic_history


def test_describe_computation_matches_expected_text_for_representative_columns():
    assert "close.pct_change(5)" in describe_computation("return_5d")
    assert "close.shift(10)" in describe_computation("log_return_10d")
    assert "100" in describe_computation("rsi_14")
    assert "rolling(20)" in describe_computation("stochastic_k_20d")
    assert describe_computation("day_of_week") == "date.dayofweek"


def test_describe_computation_returns_placeholder_for_unknown_column():
    result = describe_computation("totally_made_up_feature")

    assert result == UNKNOWN_FORMULA.format(name="totally_made_up_feature")


def test_every_real_feature_column_has_a_computation():
    history = synthetic_history(n=140)
    catalog = list_feature_columns(history)

    missing = [
        column
        for columns in catalog.values()
        for column in columns
        if describe_computation(column).startswith("No formula available")
    ]

    assert missing == [], f"missing formulas for: {missing}"
