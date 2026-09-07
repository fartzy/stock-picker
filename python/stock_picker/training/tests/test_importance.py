import numpy as np
import pandas as pd
import pytest

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.importance import ensemble_importance, model_type_importance
from stock_picker.training.model import (
    train_lightgbm,
    train_logistic_regression,
    train_neural_net,
    train_random_forest,
    train_ridge,
)


def _make_learnable_frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    label = 0.05 * np.sign(signal) + rng.normal(scale=0.005, size=n)
    return pd.DataFrame(
        {"signal": signal, "noise_feature": rng.normal(size=n), LABEL_COLUMN: label}
    )


def _trained_lightgbm():
    train_frame = _make_learnable_frame(400, seed=1)
    return train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=50)


def _trained_random_forest():
    train_frame = _make_learnable_frame(400, seed=1)
    return train_random_forest(train_frame, params={"n_estimators": 50})


def _trained_logistic_regression():
    train_frame = _make_learnable_frame(400, seed=1)
    return train_logistic_regression(train_frame)


def _trained_neural_net():
    train_frame = _make_learnable_frame(1600, seed=1)
    return train_neural_net(train_frame, params={"hidden_layer_sizes": (4,), "alpha": 1e-3, "early_stopping": False, "max_iter": 3000})


def _trained_ridge():
    train_frame = _make_learnable_frame(400, seed=1)
    return train_ridge(train_frame)


def test_lightgbm_importance_sums_to_roughly_100():
    importance = model_type_importance(_trained_lightgbm())

    assert set(importance) == {"signal", "noise_feature"}
    assert sum(importance.values()) == pytest.approx(100, abs=1.0)


def test_lightgbm_importance_ranks_the_real_signal_above_pure_noise():
    importance = model_type_importance(_trained_lightgbm())

    assert importance["signal"] > importance["noise_feature"]


def test_random_forest_importance_sums_to_roughly_100_and_ranks_signal_first():
    importance = model_type_importance(_trained_random_forest())

    assert sum(importance.values()) == pytest.approx(100, abs=1.0)
    assert importance["signal"] > importance["noise_feature"]


def test_logistic_regression_importance_sums_to_roughly_100_and_ranks_signal_first():
    importance = model_type_importance(_trained_logistic_regression())

    assert sum(importance.values()) == pytest.approx(100, abs=1.0)
    assert importance["signal"] > importance["noise_feature"]


def test_neural_net_importance_sums_to_roughly_100_and_ranks_signal_first():
    importance = model_type_importance(_trained_neural_net())

    assert sum(importance.values()) == pytest.approx(100, abs=1.0)
    assert importance["signal"] > importance["noise_feature"]


def test_ridge_importance_sums_to_roughly_100_and_ranks_signal_first():
    importance = model_type_importance(_trained_ridge())

    assert sum(importance.values()) == pytest.approx(100, abs=1.0)
    assert importance["signal"] > importance["noise_feature"]


def test_ensemble_importance_ignores_zero_weight_members():
    # A zero-weight ensemble member's importance must not leak into the
    # blended value at all, even though its own model_type_importance() is
    # real and nonzero -- weights of 0 are still a legal (if unusual) ensemble
    # composition, e.g. from a weight sweep in the composable model picker.
    lgb_model = _trained_lightgbm()
    logreg_model = _trained_logistic_regression()
    ensemble = Ensemble(members=[lgb_model, logreg_model], weights=[1.0, 0.0])

    blended = ensemble_importance(ensemble)

    assert blended == model_type_importance(lgb_model)


def test_ensemble_importance_is_the_weighted_average_of_its_members():
    lgb_model = _trained_lightgbm()
    rf_model = _trained_random_forest()
    ensemble = Ensemble(members=[lgb_model, rf_model], weights=[1.0, 1.0])

    combined = ensemble_importance(ensemble)
    lgb_importance = model_type_importance(lgb_model)
    rf_importance = model_type_importance(rf_model)

    assert combined["signal"] == pytest.approx((lgb_importance["signal"] + rf_importance["signal"]) / 2)


def test_ensemble_importance_handles_members_with_different_feature_subsets():
    train_frame = _make_learnable_frame(400, seed=1)
    only_signal = train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=20, included_features={"signal"})
    only_noise = train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=20, included_features={"noise_feature"})
    ensemble = Ensemble(members=[only_signal, only_noise], weights=[1.0, 1.0])

    combined = ensemble_importance(ensemble)

    assert set(combined) == {"signal", "noise_feature"}
    assert combined["signal"] == pytest.approx(50.0)
    assert combined["noise_feature"] == pytest.approx(50.0)
