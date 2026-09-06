import numpy as np
import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import (
    evaluate,
    feature_columns,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
)


def _make_learnable_frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(scale=0.005, size=n)
    label = 0.05 * np.sign(signal) + noise
    return pd.DataFrame({"signal": signal, "noise_feature": rng.normal(size=n), LABEL_COLUMN: label})


def test_train_lightgbm_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=50)
    metrics = evaluate(model, test_frame)

    assert model.model_type == "lightgbm"
    assert metrics.directional_accuracy > 0.9


def test_train_random_forest_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_random_forest(train_frame, params={"n_estimators": 50})
    metrics = evaluate(model, test_frame)

    assert model.model_type == "random_forest"
    assert metrics.directional_accuracy > 0.9


def test_train_logistic_regression_learns_a_clear_signal():
    # Not evaluated via evaluate() -- that helper compares np.sign() of a
    # continuous prediction against np.sign() of the continuous label, which
    # isn't meaningful for a classifier whose .predict() returns 0/1 class
    # labels, not returns. Check classification accuracy against the same
    # binarized direction the model was actually fit on.
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_logistic_regression(train_frame)
    predicted_direction = model.estimator.predict(test_frame[model.feature_names])
    actual_direction = (test_frame[LABEL_COLUMN] > 0).astype(int)

    assert model.model_type == "logistic_regression"
    assert (predicted_direction == actual_direction).mean() > 0.9


def test_feature_columns_excludes_metadata_and_label():
    frame = pd.DataFrame({"ticker": ["A"], "date": [1], "signal": [0.1], LABEL_COLUMN: [0.01]})

    assert feature_columns(frame) == ["signal"]


def test_feature_columns_also_excludes_pruned_features():
    frame = pd.DataFrame(
        {"ticker": ["A"], "date": [1], "signal": [0.1], "noise_feature": [0.2], LABEL_COLUMN: [0.01]}
    )

    assert feature_columns(frame, excluded_features={"noise_feature"}) == ["signal"]


def test_feature_columns_included_features_is_a_positive_selection():
    frame = pd.DataFrame(
        {"ticker": ["A"], "date": [1], "signal": [0.1], "other": [0.2], LABEL_COLUMN: [0.01]}
    )

    assert feature_columns(frame, included_features={"signal"}) == ["signal"]


def test_feature_columns_included_features_still_respects_exclusions():
    frame = pd.DataFrame({"signal": [0.1], "other": [0.2]})

    assert feature_columns(frame, excluded_features={"signal"}, included_features={"signal", "other"}) == ["other"]
