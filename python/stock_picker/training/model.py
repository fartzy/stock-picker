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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_picker.training.dataset import LABEL_COLUMN

NON_FEATURE_COLUMNS = {"ticker", "date", LABEL_COLUMN}

# Deliberately conservative: our current dataset is a few hundred rows across 3
# tickers with ~78 candidate features -- shallow trees and a high per-leaf sample
# floor guard against a model that just memorizes noise. Shared by both tree-based
# model types below so "same reasoning applies" is enforced by code, not just prose.
MIN_LEAF_SAMPLES = 20

LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": MIN_LEAF_SAMPLES,
    "learning_rate": 0.05,
    "verbosity": -1,
}
DEFAULT_NUM_BOOST_ROUND = 100

RANDOM_FOREST_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": MIN_LEAF_SAMPLES,
    "random_state": 0,
}
# Strong L2 regularization (low C) given ~95 candidate features over a few
# hundred rows -- a weakly-regularized logistic regression at this
# feature-to-sample ratio tends toward perfect separation on noise.
# max_iter raised well above sklearn's default (100), which often doesn't
# converge with this many features.
LOGISTIC_REGRESSION_DEFAULT_PARAMS = {
    "C": 0.1,
    "max_iter": 2000,
    "random_state": 0,
}


@dataclass
class TrainedModel:
    """One fitted model, regardless of underlying library -- what `ensemble.py`
    combines several of."""

    model_type: str  # "lightgbm" | "random_forest" | "logistic_regression"
    estimator: Any  # lgb.Booster or a fitted scikit-learn estimator
    feature_names: list[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Shared shape for both a single model's evaluate() and an ensemble's
    evaluate_ensemble() -- same three numbers either way, since the ensemble
    case just evaluates the blended prediction instead of one model's."""

    mae: float
    directional_accuracy: float
    n_test_rows: int


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


def train_logistic_regression(
    train_frame: pd.DataFrame,
    params: dict | None = None,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
) -> TrainedModel:
    """Fits on the *binarized* direction (up/down) of the label, not the
    continuous return the other two model types predict -- its coefficients
    are a genuinely different importance lens (linear/monotonic effect size
    vs. tree-based split-gain), not another return predictor. See
    `ensemble.py`'s default spec: this model type is used as a diagnostic-only
    ensemble member (weight=0.0), never blended into the actual prediction.

    Wrapped in a `Pipeline` with a median imputer, then a standard scaler:
    unlike LightGBM (handles missing values natively) and this codebase's
    RandomForestRegressor (scikit-learn added native missing-value support
    for tree ensembles in 1.4), LogisticRegression has no missing-value
    support at all and raises on any NaN -- and real feature data here is
    NaN by construction wherever a rolling window hasn't filled yet (see
    README's coverage note). Standardizing (zero mean, unit variance) before
    fitting is what makes `|coef_|` a valid cross-feature importance measure
    at all -- features here span wildly different raw scales (RSI runs
    0-100, most returns run a few percent), and an unstandardized
    coefficient's magnitude reflects that raw scale as much as it reflects
    genuine predictive effect size. The Pipeline fits the imputer's medians
    and the scaler's mean/std on the training fold only and reapplies both
    at predict time, so no separate bookkeeping is needed to keep
    fit/predict consistent.
    """
    columns = feature_columns(train_frame, excluded_features, included_features)
    direction = (train_frame[LABEL_COLUMN] > 0).astype(int)
    classifier = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classify", LogisticRegression(**{**LOGISTIC_REGRESSION_DEFAULT_PARAMS, **(params or {})})),
        ]
    )
    classifier.fit(train_frame[columns], direction)
    return TrainedModel(model_type="logistic_regression", estimator=classifier, feature_names=columns)


MODEL_TRAINERS = {
    "lightgbm": train_lightgbm,
    "random_forest": train_random_forest,
    "logistic_regression": train_logistic_regression,
}


def train_model(model_type: str, train_frame: pd.DataFrame, **kwargs) -> TrainedModel:
    return MODEL_TRAINERS[model_type](train_frame, **kwargs)


def predict(trained: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(trained.estimator.predict(frame[trained.feature_names]))


def evaluate(trained: TrainedModel, test_frame: pd.DataFrame) -> EvaluationMetrics:
    predictions = predict(trained, test_frame)
    actual = test_frame[LABEL_COLUMN].to_numpy()

    return EvaluationMetrics(
        mae=float(np.mean(np.abs(predictions - actual))),
        directional_accuracy=float(np.mean(np.sign(predictions) == np.sign(actual))),
        n_test_rows=len(test_frame),
    )
