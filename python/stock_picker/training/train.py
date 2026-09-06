"""Runs walk-forward training over the pooled dataset, logging each fold to MLflow.

MLflow here is purely for experiment tracking/observability (params, metrics, run
history) -- it is not the model registry. `main.py` persists the final fold's model
via `storage.model_store.ModelStore`, so inference never depends on an MLflow server.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from stock_picker.storage.paths import data_root
from stock_picker.training.ensemble import ModelSpec, evaluate_ensemble, train_ensemble
from stock_picker.training.splits import walk_forward_splits

DEFAULT_TRACKING_DIR = data_root() / "mlruns"
# A single LightGBM model is just a one-member ensemble -- this default keeps
# run_walk_forward's existing single-model behavior for callers that don't
# care about ensembling.
DEFAULT_SPECS: list[ModelSpec] = [ModelSpec("lightgbm")]


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
) -> list[dict]:
    """Train+evaluate an ensemble across `n_splits` walk-forward folds, logging
    each to MLflow.

    Returns fold results in chronological order: [{"fold", "model", "metrics",
    "train_rows"}, ...] ("model" is an `Ensemble`). The last entry is trained
    on the most history, and is the one `main.py` persists as the production
    model.
    """
    specs = specs if specs is not None else DEFAULT_SPECS
    _configure_mlflow(tracking_dir)
    splits = walk_forward_splits(pooled_dataset["date"], n_splits=n_splits)

    fold_results = []
    with mlflow.start_run(run_name="walk_forward"):
        _log_specs(specs, n_splits)

        for fold, (train_mask, test_mask) in enumerate(splits):
            train_frame = pooled_dataset[train_mask]
            test_frame = pooled_dataset[test_mask]

            ensemble = train_ensemble(train_frame, specs)
            metrics = evaluate_ensemble(ensemble, test_frame)

            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_param("fold", fold)
                mlflow.log_param("train_rows", len(train_frame))
                mlflow.log_metrics(metrics)

            fold_results.append(
                {
                    "fold": fold,
                    "model": ensemble,
                    "metrics": metrics,
                    "train_rows": len(train_frame),
                }
            )

    return fold_results
