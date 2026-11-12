"""Runs walk-forward training over the pooled dataset, logging each fold to MLflow.

MLflow here is purely for experiment tracking/observability (params, metrics, run
history) -- it is not the model registry. Production refits on all training
dates after validation and persists that ensemble through ModelStore.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlflow
import pandas as pd

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.storage.paths import data_root
from stock_picker.training.backtest import rank_ic, simulate_top_k
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN, iter_direction_margin_folds
from stock_picker.training.ensemble import (
    STACKED_SVM_CONSUMER_TYPES, Ensemble, ModelSpec, evaluate_ensemble, predict_ensemble, train_ensemble,
)
from stock_picker.training.model import RANK_MODEL_TYPE, EvaluationMetrics, TrainedModel, fit_stacked_svm_estimators
from stock_picker.training.splits import walk_forward_splits
from stock_picker.training.svm_stack import iter_stacked_svm_folds

DEFAULT_TRACKING_DIR = data_root() / "mlruns"
# A single LightGBM model is just a one-member ensemble -- this default keeps
# run_walk_forward's existing single-model behavior for callers that don't
# care about ensembling.
DEFAULT_SPECS: list[ModelSpec] = [ModelSpec("lightgbm")]


@dataclass
class RankEvaluationMetrics:
    """Relative-score diagnostics; a Rank score is never a return forecast."""

    rank_ic: float
    top1_avg_return: float
    top5_avg_return: float
    top20_avg_return: float
    n_test_rows: int


@dataclass
class FoldResult:
    """One validation fold; the final entry may also carry the full-data refit."""

    fold: int
    model: Ensemble
    metrics: EvaluationMetrics | RankEvaluationMetrics
    train_rows: int
    final_model: Ensemble | None = None
    final_rank_model: Ensemble | None = None


def _pair_stacked_estimators(ensemble: Ensemble, estimators: dict[str, TrainedModel]) -> None:
    """Archive only the fitted source models this head actually consumes."""
    needed = set(STACKED_SVM_COLUMNS) & {
        name for member in ensemble.members for name in member.feature_names
    }
    ensemble.stacked_svm_estimators = {name: estimators[name] for name in needed} or None
    ensemble.direction_svc = estimators.get(DIRECTION_MARGIN_COLUMN) if DIRECTION_MARGIN_COLUMN in needed else None


def _evaluate_fold(ensemble: Ensemble, frame: pd.DataFrame) -> EvaluationMetrics | RankEvaluationMetrics:
    if all(member.model_type == RANK_MODEL_TYPE for member in ensemble.members):
        scores = pd.Series(predict_ensemble(ensemble, frame), index=frame.index)
        actual, dates = frame[LABEL_COLUMN], frame["date"]
        return RankEvaluationMetrics(
            rank_ic=rank_ic(scores, actual, dates),
            top1_avg_return=simulate_top_k(scores, actual, dates, k=1)["avg_return"],
            top5_avg_return=simulate_top_k(scores, actual, dates, k=5)["avg_return"],
            top20_avg_return=simulate_top_k(scores, actual, dates, k=20)["avg_return"],
            n_test_rows=len(frame),
        )
    return evaluate_ensemble(ensemble, frame)


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
    refit_on_all_data: bool = False,
    rank_spec: ModelSpec | None = None,
) -> list[FoldResult]:
    """Train+evaluate an ensemble across chronological folds.

    When ``stack_direction_margin`` is enabled and a Fit or Rank member selects
    SVM outputs, its training rows receive strictly earlier-date predictions.
    The paired estimators are fitted
    on the entire preceding outer training period, using *all unpruned raw*
    features even when LightGBM has a narrower positive selection. Other
    member families train on those same eligible rows but never see the margin.
    ``rank_spec`` requests a separate final Rank head on that same OOF frame,
    sharing the SVM fits without putting ranking scores in the return blend.
    """
    specs = specs if specs is not None else DEFAULT_SPECS
    if not specs:
        raise ValueError("walk-forward training requires at least one model spec")
    if rank_spec is not None and (rank_spec.model_type != RANK_MODEL_TYPE or not refit_on_all_data):
        raise ValueError("a companion Rank spec requires lightgbm_rank and a full-data refit")
    consumer_specs = [*specs, *([rank_spec] if rank_spec is not None else [])]
    stacked_specs = [
        spec
        for spec in consumer_specs
        if stack_direction_margin
        and spec.model_type in STACKED_SVM_CONSUMER_TYPES
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
    fit_uses_stack = any(spec in stacked_specs for spec in specs)
    # A run has one fitted estimator per output used by Fit or Rank members.
    # Production passes a uniform pruned set to every spec; reject ambiguous
    # direct calls rather than choosing an arbitrary set of source inputs.
    source_exclusions = set(stacked_specs[0].excluded_features or ()) - set(STACKED_SVM_COLUMNS) if stacked_specs else set()
    if stacked_specs and any(
        (set(spec.excluded_features or ()) - set(STACKED_SVM_COLUMNS)) != source_exclusions
        for spec in stacked_specs[1:]
    ):
        raise ValueError("stacked-SVM Fit/Rank members must share excluded features")
    # The model-derived margin is a Fit/Rank input only. A shared
    # positive selection from the UI must not make Ridge or another member
    # consume it merely because it is present in the stacked frame.
    training_specs = [
        (
            spec
            if spec.model_type in STACKED_SVM_CONSUMER_TYPES
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
        mlflow.log_param("companion_rank", rank_spec is not None)
        mlflow.log_param("stacked_svm_outputs", ",".join(selected_outputs))

        for fold, (base_train, stacked_train, test_frame, estimators) in enumerate(folds):
            # A raw-only return head keeps every training row even when the
            # separate Rank head requested OOF inputs for its final fit.
            train_frame = (
                stacked_train if fit_uses_stack else
                pooled_dataset[pooled_dataset["date"] < test_frame["date"].min()]
            ) if estimators else base_train
            ensemble = train_ensemble(train_frame, training_specs)
            _pair_stacked_estimators(ensemble, estimators)
            metrics = _evaluate_fold(ensemble, test_frame)

            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_param("fold", fold)
                mlflow.log_param("train_rows", len(train_frame))
                mlflow.log_metrics(asdict(metrics))

            fold_results.append(
                FoldResult(fold=fold, model=ensemble, metrics=metrics, train_rows=len(train_frame))
            )

        if refit_on_all_data:
            if selected_outputs:
                # The last fold already contains earlier-date OOF outputs for
                # its training rows and for the final test block. Together
                # they cover all eligible dates through the latest close.
                stacked_final_frame = pd.concat([stacked_train, test_frame])
                if not stacked_final_frame.index.is_unique:
                    raise ValueError("full-data stacked refit requires unique OOF rows")
                final_estimators = fit_stacked_svm_estimators(
                    pooled_dataset,
                    excluded_features=source_exclusions | set(STACKED_SVM_COLUMNS),
                    outputs=selected_outputs,
                )
            else:
                stacked_final_frame = pooled_dataset
                final_estimators = {}
            final_frame = stacked_final_frame if fit_uses_stack else pooled_dataset
            final_model = train_ensemble(final_frame, training_specs)
            _pair_stacked_estimators(final_model, final_estimators)
            fold_results[-1].final_model = final_model
            if rank_spec is not None:
                rank_frame = stacked_final_frame if rank_spec in stacked_specs else pooled_dataset
                final_rank = train_ensemble(rank_frame, [rank_spec])
                _pair_stacked_estimators(final_rank, final_estimators)
                fold_results[-1].final_rank_model = final_rank
            mlflow.log_param("final_refit_rows", len(final_frame))
            mlflow.log_param("final_refit_through", str(final_frame["date"].max()))

    return fold_results
