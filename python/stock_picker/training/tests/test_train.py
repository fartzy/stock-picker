import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.storage.model_store import ModelStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN
from stock_picker.training.ensemble import ModelSpec, predict_ensemble
from stock_picker.training.model import predict
from stock_picker.training.splits import walk_forward_splits
from stock_picker.training.train import RankEvaluationMetrics, run_walk_forward


def _make_pooled_dataset(n_dates=20, n_tickers=10):
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_tickers):
        signal = rng.normal(size=n_dates)
        # Both direction classes occur even in the first tiny seed period.
        label = (0.02 + i * 0.0001) * np.where((np.arange(n_dates) + i) % 2 == 0, 1, -1)
        frames.append(
            pd.DataFrame(
                {"date": dates, "ticker": f"T{i}", "signal": signal, LABEL_COLUMN: label}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_run_walk_forward_returns_one_result_per_fold(tmp_path):
    pooled = _make_pooled_dataset()

    results = run_walk_forward(
        pooled,
        n_splits=3,
        specs=[ModelSpec("lightgbm", params={"min_data_in_leaf": 2})],
        tracking_dir=tmp_path,
        stack_direction_margin=True,
    )

    assert len(results) == 3
    assert results[-1].train_rows > results[0].train_rows
    for result in results:
        assert isinstance(result.metrics.directional_accuracy, float)
        assert set(STACKED_SVM_COLUMNS).issubset(result.model.members[0].feature_names)
        assert set(result.model.stacked_svm_estimators) == set(STACKED_SVM_COLUMNS)
        assert result.model.direction_svc is not None


@pytest.mark.parametrize("stacked", [False, True])
def test_production_refit_uses_latest_dates_and_pairs_full_history_svm(tmp_path, monkeypatch, stacked):
    from stock_picker.training import train as training_module

    pooled = _make_pooled_dataset(n_dates=30)
    newest = pooled["date"].max()
    pooled["late_feature"] = np.where(pooled["date"] >= newest - pd.Timedelta(days=5), pooled["signal"], np.nan)
    fitted = []
    source_fits = []
    actual_ensemble_fit = training_module.train_ensemble
    actual_source_fit = training_module.fit_stacked_svm_estimators

    def capture_ensemble(frame, specs):
        fitted.append(frame.copy())
        return actual_ensemble_fit(frame, specs)

    def capture_sources(frame, **kwargs):
        source_fits.append(frame["date"].max())
        return actual_source_fit(frame, **kwargs)

    monkeypatch.setattr(training_module, "train_ensemble", capture_ensemble)
    monkeypatch.setattr(training_module, "fit_stacked_svm_estimators", capture_sources)
    results = run_walk_forward(
        pooled, n_splits=3, tracking_dir=tmp_path,
        specs=[ModelSpec("lightgbm", params={"min_data_in_leaf": 2})],
        stack_direction_margin=stacked, refit_on_all_data=True,
    )

    assert len(results) == 3
    assert results[-1].final_model is not None
    assert fitted[-1]["date"].max() == newest
    assert fitted[-1]["late_feature"].notna().sum() == pooled["late_feature"].notna().sum()
    assert all(frame["date"].max() < newest for frame in fitted[:-1])
    if stacked:
        assert fitted[-1].index.is_unique
        assert np.isfinite(fitted[-1][list(STACKED_SVM_COLUMNS)].to_numpy()).all()
        assert source_fits == [newest]
        assert set(results[-1].final_model.stacked_svm_estimators) == set(STACKED_SVM_COLUMNS)
    else:
        assert not source_fits


def test_run_walk_forward_trains_fit_and_ridge_with_shared_positive_selection(tmp_path):
    pooled = _make_pooled_dataset()

    results = run_walk_forward(
        pooled,
        n_splits=3,
        specs=[
            ModelSpec(
                "lightgbm", params={"min_data_in_leaf": 2},
                included_features={"signal", DIRECTION_MARGIN_COLUMN},
            ),
            ModelSpec("ridge", included_features={"signal", DIRECTION_MARGIN_COLUMN}),
        ],
        tracking_dir=tmp_path,
        stack_direction_margin=True,
    )

    for result in results:
        assert len(result.model.members) == 2
        assert {m.model_type for m in result.model.members} == {"lightgbm", "ridge"}
        by_type = {member.model_type: member for member in result.model.members}
        assert DIRECTION_MARGIN_COLUMN in by_type["lightgbm"].feature_names
        assert by_type["ridge"].feature_names == ["signal"]


@pytest.mark.parametrize(
    "selection",
    [
        {"excluded_features": {DIRECTION_MARGIN_COLUMN}},
        {"included_features": {"signal"}},
    ],
)
def test_pruning_or_positive_selection_disables_direction_stack(tmp_path, monkeypatch, selection):
    def unexpected_stack(*_args, **_kwargs):
        raise AssertionError("the direction SVC should not be fitted")

    monkeypatch.setattr("stock_picker.training.train.iter_direction_margin_folds", unexpected_stack)
    results = run_walk_forward(
        _make_pooled_dataset(),
        n_splits=3,
        specs=[ModelSpec("lightgbm", params={"min_data_in_leaf": 2}, **selection)],
        tracking_dir=tmp_path,
        stack_direction_margin=True,
    )

    assert all(result.model.direction_svc is None for result in results)
    assert all(DIRECTION_MARGIN_COLUMN not in result.model.members[0].feature_names for result in results)


def test_svc_uses_all_unpruned_raw_inputs_and_matches_last_fit_horizon(tmp_path, monkeypatch):
    from stock_picker.training import direction_stack

    pooled = _make_pooled_dataset(n_dates=40).assign(
        other=lambda frame: frame["signal"] ** 2,
        pruned=lambda frame: frame["signal"] * -1,
    )
    actual_fit = direction_stack.train_svc_direction
    fit_horizons = []

    def capture_fit(frame, **kwargs):
        trained = actual_fit(frame, **kwargs)
        fit_horizons.append((frame["date"].max(), trained))
        return trained

    monkeypatch.setattr(direction_stack, "train_svc_direction", capture_fit)
    results = run_walk_forward(
        pooled,
        n_splits=3,
        specs=[ModelSpec(
            "lightgbm", params={"min_data_in_leaf": 2},
            excluded_features={"pruned"},
            included_features={"signal", DIRECTION_MARGIN_COLUMN},
        )],
        tracking_dir=tmp_path,
        stack_direction_margin=True,
    )

    last_train_mask, _ = walk_forward_splits(pooled["date"], n_splits=3)[-1]
    last = results[-1].model
    assert last.direction_svc is fit_horizons[-1][1]
    assert fit_horizons[-1][0] == pooled[last_train_mask]["date"].max()
    assert last.direction_svc.feature_names == ["signal", "other"]
    assert last.members[0].feature_names == ["signal", DIRECTION_MARGIN_COLUMN]
    assert results[-1].train_rows < int(last_train_mask.sum())  # early OOF warm-up is excluded


def test_research_walk_forward_is_raw_unless_stack_explicitly_enabled(tmp_path):
    results = run_walk_forward(
        _make_pooled_dataset(),
        n_splits=3,
        specs=[ModelSpec("lightgbm", params={"min_data_in_leaf": 2})],
        tracking_dir=tmp_path,
    )

    assert all(result.model.direction_svc is None for result in results)
    assert all(DIRECTION_MARGIN_COLUMN not in result.model.members[0].feature_names for result in results)


def test_only_selected_lightgbm_member_receives_margin(tmp_path):
    results = run_walk_forward(
        _make_pooled_dataset(),
        n_splits=3,
        specs=[
            ModelSpec("lightgbm", params={"min_data_in_leaf": 2},
                      included_features={"signal", DIRECTION_MARGIN_COLUMN}),
            ModelSpec("lightgbm", params={"min_data_in_leaf": 2},
                      excluded_features={DIRECTION_MARGIN_COLUMN}),
        ],
        tracking_dir=tmp_path,
        stack_direction_margin=True,
    )

    for result in results:
        first, second = result.model.members
        assert DIRECTION_MARGIN_COLUMN in first.feature_names
        assert DIRECTION_MARGIN_COLUMN not in second.feature_names
        assert result.model.direction_svc is not None


def test_research_winner_materialization_keeps_raw_comparison(monkeypatch):
    from stock_picker.training import grid_search, hour_search

    specs = [ModelSpec("lightgbm")]
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return "summary"

    monkeypatch.setattr(grid_search, "run_training", capture)
    monkeypatch.setattr(hour_search, "run_training", capture)
    grid_search._persist_raw_winner(specs)
    assert hour_search._persist_raw_winner(specs) == "summary"
    assert calls == [
        {"model_specs": specs, "stack_direction_margin": False},
        {"model_specs": specs, "stack_direction_margin": False},
    ]


def test_rank_companion_reuses_all_seven_oof_columns_and_source_fits(tmp_path, monkeypatch):
    from stock_picker.training import svm_stack
    from stock_picker.training import train as training_module

    pooled = _make_pooled_dataset(n_dates=30)
    trained = []
    folds = []
    source_calls = []
    actual_train = training_module.train_ensemble
    actual_folds = training_module.iter_stacked_svm_folds
    actual_source = training_module.fit_stacked_svm_estimators

    def capture_train(frame, specs):
        trained.append((specs[0].model_type, frame))
        return actual_train(frame, specs)

    def capture_folds(*args, **kwargs):
        for fold in actual_folds(*args, **kwargs):
            folds.append(fold)
            yield fold

    def capture_source(frame, **kwargs):
        source_calls.append((frame.index.copy(), kwargs["outputs"]))
        return actual_source(frame, **kwargs)

    monkeypatch.setattr(training_module, "train_ensemble", capture_train)
    monkeypatch.setattr(training_module, "iter_stacked_svm_folds", capture_folds)
    monkeypatch.setattr(training_module, "fit_stacked_svm_estimators", capture_source)
    monkeypatch.setattr(svm_stack, "fit_stacked_svm_estimators", capture_source)
    results = run_walk_forward(
        pooled, n_splits=3, tracking_dir=tmp_path,
        specs=[ModelSpec("lightgbm", params={"min_data_in_leaf": 2, "num_boost_round": 5})],
        rank_spec=ModelSpec("lightgbm_rank", params={"min_data_in_leaf": 2, "num_boost_round": 5}),
        stack_direction_margin=True, refit_on_all_data=True,
    )

    # Three seed fits + three outer folds + one final SVM fit; no second pass for Rank.
    assert len(source_calls) == 7
    assert all(outputs == STACKED_SVM_COLUMNS for _, outputs in source_calls)
    fit_frame, rank_frame = trained[-2][1], trained[-1][1]
    assert fit_frame is rank_frame
    expected_oof = pd.concat([folds[-1].stacked_train, folds[-1].stacked_test])
    pd.testing.assert_frame_equal(rank_frame, expected_oof)
    assert len(rank_frame) < len(pooled)
    fit, rank = results[-1].final_model, results[-1].final_rank_model
    assert [m.model_type for m in fit.members] == ["lightgbm"]
    assert [m.model_type for m in rank.members] == ["lightgbm_rank"]
    assert set(STACKED_SVM_COLUMNS).issubset(rank.members[0].feature_names)
    assert all(rank.stacked_svm_estimators[name] is fit.stacked_svm_estimators[name]
               for name in STACKED_SVM_COLUMNS)
    raw = pooled.tail(20).copy()
    manual = raw.join(svm_stack.score_stacked_svm(raw, rank.stacked_svm_estimators, STACKED_SVM_COLUMNS))
    expected = predict(rank.members[0], manual)
    restored = pickle.loads(pickle.dumps(rank))
    assert np.allclose(predict_ensemble(restored, raw), expected)
    assert np.allclose(predict_ensemble(restored, raw.assign(**dict.fromkeys(STACKED_SVM_COLUMNS, 999))), expected)


@pytest.mark.parametrize("raw_head", ["lightgbm", "lightgbm_rank"])
def test_raw_only_companion_head_keeps_full_rows_and_no_svm_artifacts(tmp_path, monkeypatch, raw_head):
    from stock_picker.training import train as training_module

    pooled = _make_pooled_dataset(n_dates=30)
    trained = []
    actual_train = training_module.train_ensemble

    def capture(frame, specs):
        trained.append((specs[0].model_type, frame))
        return actual_train(frame, specs)

    def spec(kind):
        return ModelSpec(
            kind, params={"min_data_in_leaf": 2, "num_boost_round": 5},
            included_features={"signal"} if kind == raw_head else {"signal", DIRECTION_MARGIN_COLUMN},
        )

    monkeypatch.setattr(training_module, "train_ensemble", capture)
    result = run_walk_forward(
        pooled, n_splits=3, tracking_dir=tmp_path,
        specs=[spec("lightgbm")], rank_spec=spec("lightgbm_rank"),
        stack_direction_margin=True, refit_on_all_data=True,
    )[-1]

    raw_frame = next(frame for kind, frame in trained[-2:] if kind == raw_head)
    pd.testing.assert_frame_equal(raw_frame, pooled)
    heads = {"lightgbm": result.final_model, "lightgbm_rank": result.final_rank_model}
    assert heads[raw_head].members[0].feature_names == ["signal"]
    assert heads[raw_head].stacked_svm_estimators is None
    assert heads[raw_head].direction_svc is None
    stacked_head = heads["lightgbm_rank" if raw_head == "lightgbm" else "lightgbm"]
    assert set(stacked_head.stacked_svm_estimators) == {DIRECTION_MARGIN_COLUMN}


def test_shared_svm_sources_reject_different_raw_exclusions(tmp_path):
    with pytest.raises(ValueError, match="share excluded features"):
        run_walk_forward(
            _make_pooled_dataset(), tracking_dir=tmp_path,
            specs=[ModelSpec("lightgbm", excluded_features={"signal"})],
            rank_spec=ModelSpec("lightgbm_rank"),
            stack_direction_margin=True, refit_on_all_data=True,
        )


def test_standalone_rank_uses_chronological_stack_and_saves_paired_archive(monkeypatch, tmp_path):
    from stock_picker.training import rank_model
    from stock_picker.training import train as training_module

    pooled = _make_pooled_dataset(n_dates=30)
    calls = []
    fold_metrics = []

    def walk_forward(frame, **kwargs):
        assert frame is pooled
        calls.append(kwargs)
        results = run_walk_forward(frame, n_splits=3, tracking_dir=tmp_path / "tracking", **kwargs)
        fold_metrics.extend(result.metrics for result in results)
        return results

    monkeypatch.setattr(rank_model, "UniverseStore", lambda: SimpleNamespace(active_tickers=lambda: ["TRAIN", "HOLD"]))
    monkeypatch.setattr(rank_model, "select_holdout_tickers", lambda _: {"HOLD"})
    monkeypatch.setattr(rank_model, "PriceStore", lambda: object())
    monkeypatch.setattr(rank_model, "FeatureStore", lambda: object())
    monkeypatch.setattr(rank_model, "pruned_features", lambda: {"pruned"})

    def load(tickers, *_):
        assert tickers == ["TRAIN"]
        return pooled

    monkeypatch.setattr(rank_model, "_load_pooled_dataset", load)
    monkeypatch.setattr(rank_model, "run_walk_forward", walk_forward)
    monkeypatch.setattr(
        training_module, "evaluate_ensemble",
        lambda *_: pytest.fail("unitless Rank scores cannot have return MAE or sign accuracy"),
    )
    store = ModelStore(data_dir=tmp_path / "models")
    result = rank_model.train_and_persist_rank_model(
        model_store=store,
        included_features={"signal", DIRECTION_MARGIN_COLUMN},
    )
    assert calls == [{
        "specs": [ModelSpec("lightgbm_rank", excluded_features={"pruned"},
                            included_features={"signal", DIRECTION_MARGIN_COLUMN})],
        "stack_direction_margin": True, "refit_on_all_data": True,
    }]
    restored = store.read(rank_model.RANK_MODEL_NAME)
    assert restored.members[0].feature_names == ["signal", DIRECTION_MARGIN_COLUMN]
    assert set(restored.stacked_svm_estimators) == {DIRECTION_MARGIN_COLUMN}
    assert all(isinstance(metrics, RankEvaluationMetrics) for metrics in fold_metrics)
    assert all(not hasattr(metrics, "mae") and not hasattr(metrics, "directional_accuracy")
               for metrics in fold_metrics)
    raw = pooled.tail(20)
    assert np.allclose(predict_ensemble(restored, raw), predict_ensemble(result, raw))
