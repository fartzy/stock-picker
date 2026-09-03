"""LightGBM model training and evaluation for the day-session return target."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN

NON_FEATURE_COLUMNS = {"ticker", "date", LABEL_COLUMN}

# Deliberately conservative: our current dataset is a few hundred rows across 3
# tickers with ~78 candidate features -- shallow trees and a high min_data_in_leaf
# guard against a model that just memorizes noise.
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 20,
    "learning_rate": 0.05,
    "verbosity": -1,
}
DEFAULT_NUM_BOOST_ROUND = 100


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def train_lightgbm(
    train_frame: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
) -> lgb.Booster:
    columns = feature_columns(train_frame)
    dataset = lgb.Dataset(train_frame[columns], label=train_frame[LABEL_COLUMN])
    return lgb.train(
        {**DEFAULT_PARAMS, **(params or {})}, dataset, num_boost_round=num_boost_round
    )


def evaluate(model: lgb.Booster, test_frame: pd.DataFrame) -> dict:
    columns = feature_columns(test_frame)
    predictions = model.predict(test_frame[columns])
    actual = test_frame[LABEL_COLUMN].to_numpy()

    return {
        "mae": float(np.mean(np.abs(predictions - actual))),
        "directional_accuracy": float(np.mean(np.sign(predictions) == np.sign(actual))),
        "n_test_rows": len(test_frame),
    }
