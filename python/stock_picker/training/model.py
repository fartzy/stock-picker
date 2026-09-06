"""Single-model training and evaluation for the day-session return target.

Supports more than one model family behind a common `TrainedModel` shape, so
`training/ensemble.py` can combine several without caring what's underneath
each one. See `training/ensemble.py` for combining multiple of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from stock_picker.training.dataset import LABEL_COLUMN

NON_FEATURE_COLUMNS = {"ticker", "date", LABEL_COLUMN}

# Deliberately conservative: our current dataset is a few hundred rows across 3
# tickers with ~78 candidate features -- shallow trees and a high min_data_in_leaf
# guard against a model that just memorizes noise. Same reasoning applies to the
# random forest params below (few, shallow trees; a high per-leaf sample floor).
LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 20,
    "learning_rate": 0.05,
    "verbosity": -1,
}
DEFAULT_NUM_BOOST_ROUND = 100

RANDOM_FOREST_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "random_state": 0,
}


@dataclass
class TrainedModel:
    """One fitted model, regardless of underlying library -- what `ensemble.py`
    combines several of."""

    model_type: str  # "lightgbm" | "random_forest"
    estimator: Any  # lgb.Booster or a fitted scikit-learn estimator
    feature_names: list[str] = field(default_factory=list)


def feature_columns(
    frame: pd.DataFrame,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
) -> list[str]:
    """Which columns of `frame` are actual model features.

    `included_features`, when given, is a positive selection (e.g. "only
    momentum features for this ensemble member") and takes precedence over
    `excluded_features` -- the block-list pruning already uses, and still
    applies within an inclusion set too (a pruned feature stays excluded even
    if a member's include-list names it).
    """
    excluded = excluded_features or set()
    if included_features is not None:
        return [c for c in frame.columns if c in included_features and c not in excluded]
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS and c not in excluded]


def train_lightgbm(
    train_frame: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
) -> TrainedModel:
    columns = feature_columns(train_frame, excluded_features, included_features)
    dataset = lgb.Dataset(train_frame[columns], label=train_frame[LABEL_COLUMN])
    booster = lgb.train(
        {**LIGHTGBM_DEFAULT_PARAMS, **(params or {})}, dataset, num_boost_round=num_boost_round
    )
    return TrainedModel(model_type="lightgbm", estimator=booster, feature_names=columns)


def train_random_forest(
    train_frame: pd.DataFrame,
    params: dict | None = None,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
) -> TrainedModel:
    columns = feature_columns(train_frame, excluded_features, included_features)
    forest = RandomForestRegressor(**{**RANDOM_FOREST_DEFAULT_PARAMS, **(params or {})})
    forest.fit(train_frame[columns], train_frame[LABEL_COLUMN])
    return TrainedModel(model_type="random_forest", estimator=forest, feature_names=columns)


MODEL_TRAINERS = {
    "lightgbm": train_lightgbm,
    "random_forest": train_random_forest,
}


def train_model(model_type: str, train_frame: pd.DataFrame, **kwargs) -> TrainedModel:
    return MODEL_TRAINERS[model_type](train_frame, **kwargs)


def predict(trained: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(trained.estimator.predict(frame[trained.feature_names]))


def evaluate(trained: TrainedModel, test_frame: pd.DataFrame) -> dict:
    predictions = predict(trained, test_frame)
    actual = test_frame[LABEL_COLUMN].to_numpy()

    return {
        "mae": float(np.mean(np.abs(predictions - actual))),
        "directional_accuracy": float(np.mean(np.sign(predictions) == np.sign(actual))),
        "n_test_rows": len(test_frame),
    }
