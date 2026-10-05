import pickle

import numpy as np
import pandas as pd
import pytest

from stock_picker.storage.training_config_store import ModelChoice, TrainingConfigStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import (
    ModelSpec,
    ensemble_composition,
    ensemble_feature_names,
    evaluate_ensemble,
    partition_model_specs,
    predict_ensemble,
    selected_model_specs,
    train_ensemble,
)
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN, score_direction_margin
from stock_picker.training.model import EvaluationMetrics, predict, train_model, train_svc_direction
from stock_picker.training.model import fit_stacked_svm_estimators
from stock_picker.training.svm_stack import score_stacked_svm


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


def test_ensemble_feature_names_is_the_union_across_differently_scoped_members():
    # Two members trained on disjoint feature subsets -- the union (not
    # either member alone) is what live-row building must never skip
    # computing, or predict()'s reindex would silently NaN one member's
    # inputs. See ensemble.ensemble_feature_names.
    train_frame = _make_learnable_frame(400, seed=1)
    specs = [
        ModelSpec("lightgbm", params={"min_data_in_leaf": 10}, included_features={"signal"}),
        ModelSpec("ridge", included_features={"momentum"}),
    ]
    ensemble = train_ensemble(train_frame, specs)

    assert ensemble_feature_names(ensemble) == frozenset({"signal", "momentum"})


def _ensemble_with_direction_margin():
    svc_train = _make_learnable_frame(400, seed=10)
    fit_train = _make_learnable_frame(400, seed=11)
    direction_svc = train_svc_direction(svc_train, included_features={"signal"})
    stacked_train = fit_train.assign(**{
        DIRECTION_MARGIN_COLUMN: score_direction_margin(fit_train, direction_svc)
    })
    ensemble = train_ensemble(stacked_train, [ModelSpec(
        "lightgbm", params={"min_data_in_leaf": 10},
        included_features={DIRECTION_MARGIN_COLUMN},
    )])
    ensemble.direction_svc = direction_svc
    return ensemble


def test_direction_margin_is_recomputed_from_raw_frame_for_predict_and_evaluate():
    ensemble = _ensemble_with_direction_margin()
    raw = _make_learnable_frame(50, seed=12)
    manual = raw.assign(**{
        DIRECTION_MARGIN_COLUMN: score_direction_margin(raw, ensemble.direction_svc)
    })

    expected = predict(ensemble.members[0], manual)
    assert np.allclose(predict_ensemble(ensemble, raw), expected)
    assert np.allclose(
        predict_ensemble(ensemble, raw.assign(svc_direction_margin=999.0)), expected
    )
    metrics = evaluate_ensemble(ensemble, raw)
    assert metrics.mae == pytest.approx(np.mean(np.abs(expected - raw[LABEL_COLUMN])))
    assert ensemble_feature_names(ensemble) == frozenset({"signal", DIRECTION_MARGIN_COLUMN})


def test_multiple_saved_svm_outputs_are_recomputed_at_inference():
    outputs = ("svr_oof_pred", "svc_gate_margin")
    source = _make_learnable_frame(400, seed=10)
    fit_train = _make_learnable_frame(400, seed=11)
    estimators = fit_stacked_svm_estimators(source, included_features={"signal"}, outputs=outputs)
    derived_train = score_stacked_svm(fit_train, estimators, outputs)
    stacked_train = fit_train.assign(**{name: derived_train[name] for name in outputs})
    ensemble = train_ensemble(stacked_train, [ModelSpec(
        "lightgbm", params={"min_data_in_leaf": 10}, included_features={"signal", *outputs}
    )])
    ensemble.stacked_svm_estimators = estimators
    raw = _make_learnable_frame(50, seed=12)
    derived_test = score_stacked_svm(raw, estimators, outputs)
    manual = raw.assign(**{name: derived_test[name] for name in outputs})
    expected = predict(ensemble.members[0], manual)

    assert set(outputs).issubset(ensemble.members[0].feature_names)
    assert np.allclose(predict_ensemble(ensemble, raw), expected)
    assert np.allclose(predict_ensemble(ensemble, raw.assign(**{name: 999 for name in outputs})), expected)
    assert set(ensemble_feature_names(ensemble)) == {"signal", *outputs}
    restored = pickle.loads(pickle.dumps(ensemble))
    assert np.allclose(predict_ensemble(restored, raw), expected)
    del estimators["svc_gate_margin"]
    with pytest.raises(ValueError, match="fitted SVM is missing"):
        predict_ensemble(ensemble, raw)


def test_direction_margin_requires_matching_estimator_and_raw_inputs():
    ensemble = _ensemble_with_direction_margin()
    raw = _make_learnable_frame(50, seed=12)
    ensemble.direction_svc = None
    with pytest.raises(ValueError, match="fitted SVC is missing"):
        predict_ensemble(ensemble, raw.assign(svc_direction_margin=123.0))

    ensemble.direction_svc = train_svc_direction(
        _make_learnable_frame(400, seed=10), included_features={"signal"}
    )
    with pytest.raises(ValueError, match="inputs are missing"):
        predict_ensemble(ensemble, raw.drop(columns="signal"))


def test_direction_margin_mixed_ensemble_only_adds_margin_to_lightgbm():
    svc_train = _make_learnable_frame(400, seed=10)
    fit_train = _make_learnable_frame(400, seed=11)
    raw = _make_learnable_frame(50, seed=12)
    direction_svc = train_svc_direction(svc_train, included_features={"signal"})
    stacked_train = fit_train.assign(**{
        DIRECTION_MARGIN_COLUMN: score_direction_margin(fit_train, direction_svc)
    })
    ensemble = train_ensemble(stacked_train, [
        ModelSpec("lightgbm", params={"min_data_in_leaf": 10},
                  included_features={DIRECTION_MARGIN_COLUMN}),
        ModelSpec("ridge", excluded_features={DIRECTION_MARGIN_COLUMN}),
    ])
    ensemble.direction_svc = direction_svc
    manual = raw.assign(**{DIRECTION_MARGIN_COLUMN: score_direction_margin(raw, direction_svc)})
    expected = (predict(ensemble.members[0], manual) + predict(ensemble.members[1], raw)) / 2

    assert DIRECTION_MARGIN_COLUMN not in ensemble.members[1].feature_names
    assert np.allclose(predict_ensemble(ensemble, raw), expected)
    assert np.allclose(
        predict_ensemble(ensemble, raw.assign(svc_direction_margin=999.0)), expected
    )
    assert ensemble_feature_names(ensemble) == frozenset({
        "signal", "momentum", DIRECTION_MARGIN_COLUMN
    })


def test_direction_margin_rejects_non_lightgbm_consumer_at_train_and_predict():
    svc_train = _make_learnable_frame(400, seed=10)
    fit_train = _make_learnable_frame(400, seed=11)
    raw = _make_learnable_frame(50, seed=12)
    direction_svc = train_svc_direction(svc_train, included_features={"signal"})
    stacked = fit_train.assign(**{
        DIRECTION_MARGIN_COLUMN: score_direction_margin(fit_train, direction_svc)
    })
    with pytest.raises(ValueError, match="only supported for LightGBM members"):
        train_ensemble(stacked, [ModelSpec("ridge", included_features={DIRECTION_MARGIN_COLUMN})])

    # Simulate a malformed archived ensemble that bypassed the training guard.
    ensemble = _ensemble_with_direction_margin()
    ensemble.members.append(train_model("ridge", stacked, included_features={DIRECTION_MARGIN_COLUMN}))
    ensemble.weights.append(1.0)
    with pytest.raises(ValueError, match="only supported for LightGBM members"):
        predict_ensemble(ensemble, raw.assign(svc_direction_margin=999.0))


def test_partition_model_specs_none_means_default_return_and_rank():
    predictive, wants_rank = partition_model_specs(None)

    assert predictive is None
    assert wants_rank is True


def test_partition_model_specs_peels_rank_out_of_the_return_blend():
    specs = [ModelSpec("lightgbm", weight=1.0), ModelSpec("lightgbm_rank", weight=1.0)]

    predictive, wants_rank = partition_model_specs(specs)

    assert [(s.model_type, s.weight) for s in predictive] == [("lightgbm", 1.0)]
    assert wants_rank is True


def test_partition_model_specs_rank_only_still_trains_default_return():
    predictive, wants_rank = partition_model_specs([ModelSpec("lightgbm_rank")])

    assert predictive is None
    assert wants_rank is True


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
