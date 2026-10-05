"""Runs walk-forward training over the pooled dataset, logging each fold to MLflow.

MLflow here is purely for experiment tracking/observability (params, metrics, run
history) -- it is not the model registry. `main.py` persists the final fold's model
via `storage.model_store.ModelStore`, so inference never depends on an MLflow server.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlflow
import pandas as pd

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.storage.paths import data_root
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN, iter_direction_margin_folds
from stock_picker.training.ensemble import Ensemble, ModelSpec, evaluate_ensemble, train_ensemble
from stock_picker.training.model import EvaluationMetrics
from stock_picker.training.splits import walk_forward_splits
from stock_picker.training.svm_stack import iter_stacked_svm_folds

DEFAULT_TRACKING_DIR = data_root() / "mlruns"
# A single LightGBM model is just a one-member ensemble -- this default keeps
# run_walk_forward's existing single-model behavior for callers that don't
# care about ensembling.
DEFAULT_SPECS: list[ModelSpec] = [ModelSpec("lightgbm")]


@dataclass
class FoldResult:
    """One walk-forward fold's outcome. Chronological order; the last entry
    is trained on the most history, and is the one main.py persists as the
    production model."""

    fold: int
    model: Ensemble
    metrics: EvaluationMetrics
    train_rows: int


def _configure_mlflow(tracking_dir: Path) -> None:
    # MLflow's raw filesystem tracking backend is deprecated/maintenance-mode as of
    # MLflow 3.x -- sqlite is their current recommendation for a local, serverless
    # backend, so this is still "local deployment mode," just via a local DB file
    # instead of a bare directory.
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir.resolve()}/mlflow.db")
    mlflow.set_experiment("day_session_return")


def _log_specs(specs: list[ModelSpec], n_splits: int) -> None:
    params = {"n_splits": n_splits, "ensemble_size": len(specs)}
    for i, spec in enumerate(specs):
        params[f"model_{i}_type"] = spec.model_type
        params[f"model_{i}_weight"] = spec.weight
        for key, value in (spec.params or {}).items():
            params[f"model_{i}_{key}"] = value
    mlflow.log_params(params)


def run_walk_forward(
    pooled_dataset: pd.DataFrame,
    n_splits: int = 4,
    specs: list[ModelSpec] | None = None,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    stack_direction_margin: bool = False,
) -> list[FoldResult]:
    """Train+evaluate an ensemble across chronological folds.

    When ``stack_direction_margin`` is enabled and a LightGBM member selects
    SVM outputs, its training rows receive strictly earlier-date predictions.
    The paired estimators are fitted
    on the entire preceding outer training period, using *all unpruned raw*
    features even when LightGBM has a narrower positive selection. Other
    member families train on those same eligible rows but never see the margin.
    """
    specs = specs if specs is not None else DEFAULT_SPECS
    stacked_specs = [
        spec
        for spec in specs
        if stack_direction_margin
        and spec.model_type == "lightgbm"
        and any(
            name not in (spec.excluded_features or set())
            and (spec.included_features is None or name in spec.included_features)
            for name in STACKED_SVM_COLUMNS
        )
    ]
    selected_outputs = tuple(
        name for name in STACKED_SVM_COLUMNS
        if any(
            name not in (spec.excluded_features or set())
            and (spec.included_features is None or name in spec.included_features)
            for spec in stacked_specs
        )
    )
    # A run has one fitted estimator per output used by LightGBM members.
    # Production passes a uniform pruned set to every spec; reject ambiguous
    # direct calls rather than choosing an arbitrary set of source inputs.
    source_exclusions = set(stacked_specs[0].excluded_features or ()) - set(STACKED_SVM_COLUMNS) if stacked_specs else set()
    if stacked_specs and any(
        (set(spec.excluded_features or ()) - set(STACKED_SVM_COLUMNS)) != source_exclusions
        for spec in stacked_specs[1:]
    ):
        raise ValueError("stacked-SVM LightGBM members must share excluded features")
    # The model-derived margin is a Fit/LightGBM input only. A shared
    # positive selection from the UI must not make Ridge or another member
    # consume it merely because it is present in the stacked frame.
    training_specs = [
        (
            spec
            if spec.model_type == "lightgbm"
            else replace(
                spec,
                excluded_features=set(spec.excluded_features or ()) | set(STACKED_SVM_COLUMNS),
                included_features=(
                    None if spec.included_features is None
                    else set(spec.included_features) - set(STACKED_SVM_COLUMNS)
                ),
            )
        )
        for spec in specs
    ]
    _configure_mlflow(tracking_dir)
    if selected_outputs == (DIRECTION_MARGIN_COLUMN,):
        source_exclusions = set(stacked_specs[0].excluded_features or ()) | set(STACKED_SVM_COLUMNS)
        stacked_folds = iter_direction_margin_folds(
            pooled_dataset,
            excluded_features=source_exclusions,
            n_splits=n_splits,
        )
        folds = (
            (fold.baseline_train, fold.stacked_train, fold.stacked_test,
             {DIRECTION_MARGIN_COLUMN: fold.direction_svc})
            for fold in stacked_folds
        )
    elif selected_outputs:
        stacked_folds = iter_stacked_svm_folds(
            pooled_dataset,
            outputs=selected_outputs,
            excluded_features=stacked_specs[0].excluded_features,
            n_splits=n_splits,
        )
        folds = (
            (fold.baseline_train, fold.stacked_train, fold.stacked_test, fold.estimators)
            for fold in stacked_folds
        )
    else:
        splits = walk_forward_splits(pooled_dataset["date"], n_splits=n_splits)
        folds = (
            (pooled_dataset[train_mask], pooled_dataset[train_mask], pooled_dataset[test_mask], {})
            for train_mask, test_mask in splits
        )

    fold_results = []
    with mlflow.start_run(run_name="walk_forward"):
        _log_specs(specs, n_splits)
        mlflow.log_param("stacked_svm_outputs", ",".join(selected_outputs))

        for fold, (base_train, stacked_train, test_frame, estimators) in enumerate(folds):
            train_frame = stacked_train if estimators else base_train
            ensemble = train_ensemble(train_frame, training_specs)
            ensemble.stacked_svm_estimators = estimators or None
            ensemble.direction_svc = estimators.get(DIRECTION_MARGIN_COLUMN)
            metrics = evaluate_ensemble(ensemble, test_frame)

            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_param("fold", fold)
                mlflow.log_param("train_rows", len(train_frame))
                mlflow.log_metrics(asdict(metrics))

            fold_results.append(
                FoldResult(fold=fold, model=ensemble, metrics=metrics, train_rows=len(train_frame))
            )

    return fold_results
