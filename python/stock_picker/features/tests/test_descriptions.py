from stock_picker.features.catalog import list_feature_columns
from stock_picker.features.descriptions import UNKNOWN_DESCRIPTION, describe_feature
from stock_picker.features.tests.fixtures import synthetic_history


def test_describe_feature_matches_expected_text_for_representative_columns():
    assert "5 trading days" in describe_feature("return_5d")
    assert "Log return" in describe_feature("log_return_10d")
    assert describe_feature("consecutive_day_streak").startswith("Signed count")
    assert "mean-reversion" in describe_feature("rsi_2")
    assert "mean-reversion" not in describe_feature("rsi_14")
    assert "5" in describe_feature("stochastic_k_5d")
    assert describe_feature("day_of_week").startswith("Day of the week")


def test_describe_feature_returns_placeholder_for_unknown_column():
    result = describe_feature("totally_made_up_feature")

    assert result == UNKNOWN_DESCRIPTION.format(name="totally_made_up_feature")


def test_every_real_feature_column_has_a_description():
    history = synthetic_history(n=140)
    catalog = list_feature_columns(history)

    missing = [
        column
        for columns in catalog.values()
        for column in columns
        if describe_feature(column).startswith("No description available")
    ]

    assert missing == [], f"missing descriptions for: {missing}"
