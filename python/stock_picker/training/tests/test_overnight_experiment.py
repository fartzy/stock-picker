"""Controlled overnight feature comparison and saved-row contract."""

import numpy as np
import pandas as pd
import pytest

from stock_picker.training.overnight_dataset import LABEL_COLUMN
from stock_picker.training.overnight_experiment import (
    CORE_COLUMNS, EXISTING_COLUMNS, VARIANTS, evaluate_variants,
)
from stock_picker.training.overnight_model import MODEL_FEATURE_COLUMNS


def test_variants_keep_original_inputs_and_emphasize_price_move_plus_fit():
    assert CORE_COLUMNS == ("assumed_day_return", "day_fit_predicted_return")
    assert len(EXISTING_COLUMNS) == 13
    assert len(MODEL_FEATURE_COLUMNS) == 15
    assert "prior_return_5d" not in EXISTING_COLUMNS
    weighted = VARIANTS[-1]
    assert weighted.name == "weighted_15"
    weights = weighted.params()["feature_contri"]
    assert len(weights) == len(MODEL_FEATURE_COLUMNS)
    assert [weights[MODEL_FEATURE_COLUMNS.index(column)] for column in CORE_COLUMNS] == [3.0, 3.0]
    assert all(weight == 1.0 for column, weight in zip(MODEL_FEATURE_COLUMNS, weights)
               if column not in CORE_COLUMNS)


def test_every_variant_uses_identical_chronological_test_rows():
    dates = pd.bdate_range("2026-01-05", periods=18)
    rows = []
    for day_index, session in enumerate(dates):
        for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
            row = {column: 0.001 * (day_index + ticker_index) for column in MODEL_FEATURE_COLUMNS}
            row.update({
                "ticker": ticker, "date": session,
                "assumed_close": 20.0 + ticker_index + day_index * 0.1,
                "assumed_day_return": 0.002 * (day_index % 5 - 2),
                "day_fit_predicted_return": 0.003 * (day_index % 4 - 1),
                LABEL_COLUMN: 0.004 * ((day_index + ticker_index) % 5 - 2),
            })
            row.pop("fit_minus_assumed_day_return")
            rows.append(row)
    predictions, summary = evaluate_variants(
        pd.DataFrame(rows), n_splits=2, rounds=5, params={"min_data_in_leaf": 2, "num_threads": 1},
    )
    assert len(predictions) == 4 * 2 * 6 * 3
    assert len(summary) == 8
    assert predictions.groupby(["variant", "fold"]).size().nunique() == 1
    assert (predictions["train_through"] < predictions["date"]).all()
    assert predictions["actual_next_open"].to_numpy() == pytest.approx(
        predictions["assumed_close"].to_numpy() * (1 + predictions["actual_gap"].to_numpy())
    )
    assert np.isfinite(predictions["predicted_gap"]).all()
    assert summary.groupby("fold")["zero_gap_mae"].nunique().eq(1).all()
