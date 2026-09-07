from stock_picker.features.catalog import list_feature_columns
from stock_picker.features.examples import UNKNOWN_EXAMPLE, feature_example
from stock_picker.features.tests.fixtures import synthetic_history


def test_feature_example_matches_expected_text_for_representative_columns():
    assert "5 days" in feature_example("return_5d")
    assert "+0.02" in feature_example("overnight_gap")
    assert feature_example("consecutive_day_streak").startswith("Three straight")
    assert "10" in feature_example("stochastic_k_10d")
    assert feature_example("day_of_week").startswith("A Wednesday")


def test_feature_example_returns_placeholder_for_unknown_column():
    result = feature_example("totally_made_up_feature")

    assert result == UNKNOWN_EXAMPLE.format(name="totally_made_up_feature")


def test_every_real_feature_column_has_an_example():
    history = synthetic_history(n=140)
    catalog = list_feature_columns(history)

    missing = [
        column
        for columns in catalog.values()
        for column in columns
        if feature_example(column).startswith("No example available")
    ]

    assert missing == [], f"missing examples for: {missing}"
