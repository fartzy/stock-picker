import numpy as np
import pandas as pd
import pytest

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN
from stock_picker.training.ensemble import ModelSpec
from stock_picker.training.splits import walk_forward_splits
from stock_picker.training.train import run_walk_forward


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
