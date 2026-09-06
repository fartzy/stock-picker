import numpy as np
import pandas as pd

from stock_picker.storage.training_config_store import ModelChoice, TrainingConfigStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import (
    ModelSpec,
    ensemble_composition,
    evaluate_ensemble,
    predict_ensemble,
    selected_model_specs,
    train_ensemble,
)
from stock_picker.training.model import EvaluationMetrics


def _make_learnable_frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    momentum = rng.normal(size=n)
    label = 0.05 * np.sign(signal) + rng.normal(scale=0.005, size=n)
    return pd.DataFrame({"signal": signal, "momentum": momentum, LABEL_COLUMN: label})


def test_train_ensemble_with_two_differently_typed_members_predicts_sanely():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)
    specs = [
        ModelSpec("lightgbm", params={"min_data_in_leaf": 10}),
        ModelSpec("random_forest", params={"n_estimators": 50}),
    ]

    ensemble = train_ensemble(train_frame, specs)
    predictions = predict_ensemble(ensemble, test_frame)

    assert len(ensemble.members) == 2
    assert predictions.shape == (len(test_frame),)
    metrics = evaluate_ensemble(ensemble, test_frame)
    assert metrics.directional_accuracy > 0.9


def test_a_member_with_included_features_only_sees_those_columns():
    train_frame = _make_learnable_frame(400, seed=1)
    specs = [ModelSpec("lightgbm", params={"min_data_in_leaf": 10}, included_features={"signal"})]

    ensemble = train_ensemble(train_frame, specs)

    assert ensemble.members[0].feature_names == ["signal"]


def test_weights_change_the_blended_prediction():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(50, seed=3)
    specs_equal = [
        ModelSpec("lightgbm", params={"min_data_in_leaf": 10}, weight=1.0),
        ModelSpec("random_forest", params={"n_estimators": 50}, weight=1.0),
    ]
    specs_skewed = [
        ModelSpec("lightgbm", params={"min_data_in_leaf": 10}, weight=10.0),
        ModelSpec("random_forest", params={"n_estimators": 50}, weight=0.01),
    ]

    equal_ensemble = train_ensemble(train_frame, specs_equal)
    skewed_ensemble = train_ensemble(train_frame, specs_skewed)

    equal_predictions = predict_ensemble(equal_ensemble, test_frame)
    skewed_predictions = predict_ensemble(skewed_ensemble, test_frame)

    assert not np.allclose(equal_predictions, skewed_predictions)


def test_evaluate_ensemble_returns_the_same_metric_shape_as_a_single_model():
    train_frame = _make_learnable_frame(400, seed=1)
    test_frame = _make_learnable_frame(200, seed=2)
    ensemble = train_ensemble(train_frame, [ModelSpec("lightgbm", params={"min_data_in_leaf": 10})])

    metrics = evaluate_ensemble(ensemble, test_frame)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.n_test_rows == len(test_frame)


def test_ensemble_composition_reports_each_members_type_weight_and_feature_count():
    train_frame = _make_learnable_frame(400, seed=1)
    specs = [
        ModelSpec("lightgbm", params={"min_data_in_leaf": 10}, weight=1.0),
        ModelSpec("random_forest", params={"n_estimators": 50}, weight=0.5),
    ]
    ensemble = train_ensemble(train_frame, specs)

    composition = ensemble_composition(ensemble)

    assert [m.model_type for m in composition] == ["lightgbm", "random_forest"]
    assert [m.weight for m in composition] == [1.0, 0.5]
    # Both members see the same two feature columns here -- signal/momentum.
    assert all(m.feature_count == 2 for m in composition)


def test_selected_model_specs_returns_none_when_nothing_persisted(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "stock_picker.training.ensemble.TrainingConfigStore",
        lambda: TrainingConfigStore(data_dir=tmp_path),
    )

    assert selected_model_specs() is None


def test_selected_model_specs_reflects_persisted_choices(monkeypatch, tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)
    store.write_model_choices([ModelChoice("lightgbm", weight=2.0)])
    monkeypatch.setattr("stock_picker.training.ensemble.TrainingConfigStore", lambda: store)

    specs = selected_model_specs()

    assert [(s.model_type, s.weight) for s in specs] == [("lightgbm", 2.0)]
