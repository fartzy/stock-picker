import numpy as np
import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import (
    STACKED_SVM_COLUMNS,
    attach_stacked_svm_columns,
    decision_scores,
    evaluate,
    feature_columns,
    fit_stacked_svm_estimators,
    train_lightgbm,
    train_lightgbm_rank,
    train_logistic_regression,
    train_neural_net,
    train_random_forest,
    train_ridge,
    train_svc_bottom_quintile,
    train_svc_direction,
    train_svc_down_gate,
    train_svc_gate,
    train_svc_strong_up,
    train_svc_top_quintile,
    train_svr,
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


def test_train_lightgbm_is_reproducible_across_runs():
    # feature_fraction/bagging_fraction in LIGHTGBM_DEFAULT_PARAMS draw a
    # random subsample every training run -- without a fixed seed, two
    # trains on identical data produce different models (and therefore
    # different holdout numbers), which reads as a real config/feature
    # improvement or regression when it's actually just RNG noise.
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    first = train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=50)
    second = train_lightgbm(train_frame, params={"min_data_in_leaf": 10}, num_boost_round=50)

    first_predictions = first.estimator.predict(test_frame[feature_columns(test_frame)])
    second_predictions = second.estimator.predict(test_frame[feature_columns(test_frame)])
    assert np.array_equal(first_predictions, second_predictions)


def test_train_lightgbm_rank_orders_same_day_names():
    dates = pd.to_datetime(["2026-01-02"] * 40 + ["2026-01-03"] * 40)
    rng = np.random.default_rng(0)
    signal = rng.normal(size=80)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "date": dates,
            LABEL_COLUMN: 0.04 * signal + rng.normal(scale=0.002, size=80),
        }
    )
    train = frame.iloc[:60]
    test = frame.iloc[60:]

    model = train_lightgbm_rank(train, params={"min_data_in_leaf": 5}, num_boost_round=40)
    pred = pd.Series(model.estimator.predict(test[feature_columns(test)]), index=test.index)
    ic = pred.corr(test[LABEL_COLUMN], method="spearman")

    assert model.model_type == "lightgbm_rank"
    assert ic > 0.5


def test_train_random_forest_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_random_forest(train_frame, params={"n_estimators": 50})
    metrics = evaluate(model, test_frame)

    assert model.model_type == "random_forest"
    assert metrics.directional_accuracy > 0.9


_NEURAL_NET_TEST_PARAMS = {
    "hidden_layer_sizes": (4,),
    "alpha": 1e-3,
    "early_stopping": False,
    "max_iter": 3000,
}


def test_train_neural_net_learns_a_clear_signal():
    # More rows than the other trainers' equivalent test -- a small MLP
    # needs more data than a shallow tree/linear model to reliably converge
    # on a solution dominated by the real signal rather than noise.
    train_frame = _make_learnable_frame(1600, seed=1)
    test_frame = _make_learnable_frame(800, seed=2)

    model = train_neural_net(train_frame, params=_NEURAL_NET_TEST_PARAMS)
    metrics = evaluate(model, test_frame)

    assert model.model_type == "neural_net"
    # Lower bar than the tree/linear trainers' equivalent test (> 0.9) --
    # gradient-based training on a tiny network is a noisier fit than a
    # shallow tree or closed-form-ish linear solve, even on a clean signal.
    assert metrics.directional_accuracy > 0.7


def test_train_neural_net_handles_missing_values():
    # MLPRegressor has no native NaN support (raises on any missing value),
    # so this only passes if the Pipeline's impute step is actually wired in
    # -- real feature data is NaN by construction wherever a rolling window
    # hasn't filled yet.
    train_frame = _make_learnable_frame(1600, seed=1)
    train_frame.loc[train_frame.index[:20], "signal"] = None

    model = train_neural_net(train_frame, params=_NEURAL_NET_TEST_PARAMS)

    assert model.model_type == "neural_net"


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


def test_train_ridge_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_ridge(train_frame)
    metrics = evaluate(model, test_frame)

    assert model.model_type == "ridge"
    assert metrics.directional_accuracy > 0.9


def test_train_ridge_handles_missing_values():
    # Ridge has no native NaN support (raises on any missing value), so this
    # only passes if the Pipeline's impute step is actually wired in -- real
    # feature data is NaN by construction wherever a rolling window hasn't
    # filled yet.
    train_frame = _make_learnable_frame(400, seed=1)
    train_frame.loc[train_frame.index[:20], "signal"] = None

    model = train_ridge(train_frame)

    assert model.model_type == "ridge"


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


def test_train_svr_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svr(train_frame)
    metrics = evaluate(model, test_frame)

    assert model.model_type == "svr"
    assert metrics.directional_accuracy > 0.9


def test_train_svr_handles_missing_values():
    # LinearSVR has no native NaN support (raises on any missing value), so
    # this only passes if the Pipeline's impute step is actually wired in.
    train_frame = _make_learnable_frame(400, seed=1)
    train_frame.loc[train_frame.index[:20], "signal"] = None

    model = train_svr(train_frame)

    assert model.model_type == "svr"


def test_train_svc_direction_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svc_direction(train_frame)
    predicted = model.estimator.predict(test_frame[model.feature_names])
    actual = (test_frame[LABEL_COLUMN] > 0).astype(int)

    assert model.model_type == "svc_direction"
    assert (predicted == actual).mean() > 0.9


def test_train_svc_gate_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svc_gate(train_frame)
    predicted = model.estimator.predict(test_frame[model.feature_names])
    actual = (test_frame[LABEL_COLUMN] > 0.005).astype(int)

    assert model.model_type == "svc_gate"
    assert (predicted == actual).mean() > 0.9


def test_train_svc_down_gate_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svc_down_gate(train_frame)
    predicted = model.estimator.predict(test_frame[model.feature_names])
    actual = (test_frame[LABEL_COLUMN] < -0.005).astype(int)

    assert model.model_type == "svc_down_gate"
    assert (predicted == actual).mean() > 0.9


def test_train_svc_strong_up_learns_a_clear_signal():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svc_strong_up(train_frame)
    predicted = model.estimator.predict(test_frame[model.feature_names])
    actual = (test_frame[LABEL_COLUMN] > 0.01).astype(int)

    assert model.model_type == "svc_strong_up"
    assert (predicted == actual).mean() > 0.9


def test_train_svc_top_quintile_ranks_the_right_tail():
    # Top-quintile is a 20% class -- majority-class accuracy is already 0.80,
    # so the stacking-relevant check is that the margin ranks true top-20
    # names above the rest.
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svc_top_quintile(train_frame)
    scores = decision_scores(model, test_frame)
    actual = test_frame[LABEL_COLUMN].rank(pct=True) >= 0.8

    assert model.model_type == "svc_top_quintile"
    assert scores[actual.to_numpy()].mean() > scores[~actual.to_numpy()].mean()


def test_train_svc_bottom_quintile_ranks_the_right_tail():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)

    model = train_svc_bottom_quintile(train_frame)
    scores = decision_scores(model, test_frame)
    actual = test_frame[LABEL_COLUMN].rank(pct=True) <= 0.2

    assert model.model_type == "svc_bottom_quintile"
    assert scores[actual.to_numpy()].mean() > scores[~actual.to_numpy()].mean()


def test_attach_stacked_svm_columns_adds_the_seven_adr_0021_columns():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)
    estimators = fit_stacked_svm_estimators(train_frame)
    stacked = attach_stacked_svm_columns(test_frame, estimators)

    assert list(STACKED_SVM_COLUMNS) == [
        "svr_oof_pred",
        "svc_direction_margin",
        "svc_gate_margin",
        "svc_down_gate_margin",
        "svc_strong_up_margin",
        "svc_top_quintile_margin",
        "svc_bottom_quintile_margin",
    ]
    assert set(estimators) == set(STACKED_SVM_COLUMNS)
    for column in STACKED_SVM_COLUMNS:
        assert column in stacked.columns
        assert stacked[column].notna().all()
    assert len(stacked) == len(test_frame)
    # Original columns stay put -- stacking appends, it does not rewrite
    # the open-known row.
    assert (stacked["signal"] == test_frame["signal"]).all()
