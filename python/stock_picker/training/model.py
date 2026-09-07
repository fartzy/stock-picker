"""Single-model training and evaluation for the day-session return target.

Supports more than one model family behind a common `TrainedModel` shape, so
`training/ensemble.py` can combine several without caring what's underneath
each one. See `training/ensemble.py` for combining multiple of these.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
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
    # Tuned against 4-fold walk-forward validation accuracy over the full
    # 500-ticker universe (see training/tune_experiment.py) -- deeper than
    # the original conservative defaults, which were sized for a dataset a
    # few hundred rows across 3 tickers, not this one's ~113k rows. Retuned
    # after adding the overnight-gap interaction features (gap_volume_
    # interaction, gap_streak): 58.2% holdout accuracy and 79.2% hit rate at
    # the 0.5% threshold, both up from the prior config's 57.7%/77.3%.
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 30,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    # LightGBM's "0 means default OpenMP thread count" hasn't been reliably
    # detecting all cores in practice -- pin it explicitly instead, same
    # reasoning as RandomForestRegressor's n_jobs below.
    "num_threads": os.cpu_count() or 4,
}
DEFAULT_NUM_BOOST_ROUND = 100

RANDOM_FOREST_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": MIN_LEAF_SAMPLES,
    "random_state": 0,
    # sklearn defaults to a single core; this dataset is large enough now
    # (500-ticker universe) that leaving 13 of 14 cores idle noticeably
    # slows every training run for no reason.
    "n_jobs": -1,
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
# Small architecture and strong L2 (alpha) for the same reason as the other
# trainers' conservative defaults: day-session return is dominated by noise
# at this feature set, so a wide/deep net just memorizes it. early_stopping
# holds out its own internal validation slice and halts once that stops
# improving, rather than always running to max_iter regardless of overfit.
NEURAL_NET_DEFAULT_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-2,
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
    "random_state": 0,
}
# alpha=1.0 is sklearn's own default -- a reasonable starting point given
# production's feature-to-sample ratio is far more favorable here (~90
# features over ~113k rows) than logistic_regression's original few-hundred-
# row concern above; the label is still noisy day-session return, so some
# regularization stays warranted, just not as aggressive.
RIDGE_DEFAULT_PARAMS = {
    "alpha": 1.0,
    "random_state": 0,
}


@dataclass
class TrainedModel:
    """One fitted model, regardless of underlying library -- what `ensemble.py`
    combines several of."""

    model_type: str  # "lightgbm" | "random_forest" | "logistic_regression" | "neural_net" | "ridge"
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
    vs. tree-based split-gain), not another return predictor. Its binary
    output isn't compatible with `Ensemble`'s weighted-average blending of
    continuous predictions, so it's fit and persisted standalone (see
    `training/main.py`'s `run_training`) rather than as an ensemble member.

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


def train_neural_net(
    train_frame: pd.DataFrame,
    params: dict | None = None,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
) -> TrainedModel:
    """Predicts the same continuous day-session return LightGBM/RandomForest
    do (unlike train_logistic_regression's binarized direction), so it fits
    Ensemble's weighted-average blend natively -- no standalone-fit
    workaround needed.

    Wrapped in the same impute+scale Pipeline as train_logistic_regression,
    for the same reason: MLPRegressor has no missing-value support (raises
    on any NaN, and real feature data here is NaN by construction wherever a
    rolling window hasn't filled yet) and gradient-based training on
    unscaled inputs spanning wildly different raw ranges (RSI 0-100 vs.
    returns ~0-5%) converges poorly or not at all.
    """
    columns = feature_columns(train_frame, excluded_features, included_features)
    regressor = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("regress", MLPRegressor(**{**NEURAL_NET_DEFAULT_PARAMS, **(params or {})})),
        ]
    )
    regressor.fit(train_frame[columns], train_frame[LABEL_COLUMN])
    return TrainedModel(model_type="neural_net", estimator=regressor, feature_names=columns)


def train_ridge(
    train_frame: pd.DataFrame,
    params: dict | None = None,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
) -> TrainedModel:
    """Predicts the same continuous day-session return the other predictive
    trainers do, so it fits Ensemble's weighted-average blend natively. A
    linear model is a structurally different lens than the tree-based
    (LightGBM/RandomForest) and gradient-based (neural_net) trainers above --
    worth testing empirically as its own ensemble candidate, same as each of
    those was (see training/tune_experiment.py).

    Wrapped in the same impute+scale Pipeline as train_neural_net, for the
    same two reasons: Ridge has no native missing-value support, and L2
    regularization penalizes every coefficient's magnitude uniformly --
    meaningless unless every feature is on the same scale first (RSI runs
    0-100, most returns run a few percent).
    """
    columns = feature_columns(train_frame, excluded_features, included_features)
    regressor = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("regress", Ridge(**{**RIDGE_DEFAULT_PARAMS, **(params or {})})),
        ]
    )
    regressor.fit(train_frame[columns], train_frame[LABEL_COLUMN])
    return TrainedModel(model_type="ridge", estimator=regressor, feature_names=columns)


MODEL_TRAINERS = {
    "lightgbm": train_lightgbm,
    "random_forest": train_random_forest,
    "logistic_regression": train_logistic_regression,
    "neural_net": train_neural_net,
    "ridge": train_ridge,
}

# The model types that predict the continuous day-session return and can
# therefore be blended into an Ensemble. logistic_regression predicts a
# binary direction instead -- a different unit that can't be weighted-
# averaged with these, so it's fit standalone (see training/main.py) and
# deliberately excluded from this list, which is what the composable
# model-type picker in the UI offers.
PREDICTIVE_MODEL_TYPES = ["lightgbm", "random_forest", "neural_net", "ridge"]


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
