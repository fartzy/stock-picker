"""Leakage-safe, chronological SVM outputs for LightGBM Fit training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import (
    TrainedModel,
    decision_scores,
    fit_stacked_svm_estimators,
    predict,
)
from stock_picker.training.splits import walk_forward_splits

DEFAULT_OOF_SEED_SPLITS = 3


@dataclass(frozen=True)
class StackedSvmFold:
    baseline_train: pd.DataFrame
    stacked_train: pd.DataFrame
    stacked_test: pd.DataFrame
    estimators: dict[str, TrainedModel]


def score_stacked_svm(
    frame: pd.DataFrame, estimators: dict[str, TrainedModel], outputs: tuple[str, ...]
) -> pd.DataFrame:
    """Compute requested outputs from their matching saved estimators, never from supplied values."""
    columns: dict[str, pd.Series] = {}
    for name in outputs:
        estimator = estimators.get(name)
        if estimator is None:
            raise ValueError(f"LightGBM expects {name} but its fitted SVM is missing")
        if name not in STACKED_SVM_COLUMNS:
            raise ValueError(f"unknown stacked SVM output: {name}")
        expected_type = "svr" if name == "svr_oof_pred" else name.removesuffix("_margin")
        if estimator.model_type != expected_type:
            raise ValueError(f"{name} requires a fitted {expected_type} estimator")
        missing = set(estimator.feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} inputs are missing: {sorted(missing)}")
        if set(estimator.feature_names) & set(STACKED_SVM_COLUMNS):
            raise ValueError(f"{name} SVM cannot consume stacked SVM outputs")
        if frame[estimator.feature_names].isna().all(axis=1).any():
            raise ValueError(f"{name} inputs are all missing for at least one row")
        values = np.asarray(
            predict(estimator, frame) if name == "svr_oof_pred" else decision_scores(estimator, frame),
            dtype=float,
        )
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError(f"{name} values must be finite and row-aligned")
        columns[name] = pd.Series(values, index=frame.index)
    return pd.DataFrame(columns, index=frame.index)


def _fit_and_score(
    fit_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    outputs: tuple[str, ...],
    excluded_features: set[str] | None,
) -> tuple[dict[str, TrainedModel], pd.DataFrame]:
    if fit_frame.empty or score_frame.empty or fit_frame["date"].max() >= score_frame["date"].min():
        raise ValueError("stacked SVM requires strictly earlier fitting dates")
    estimators = fit_stacked_svm_estimators(
        fit_frame,
        excluded_features=set(excluded_features or ()) | set(STACKED_SVM_COLUMNS),
        outputs=outputs,
    )
    return estimators, score_stacked_svm(score_frame, estimators, outputs)


def iter_stacked_svm_folds(
    pooled_dataset: pd.DataFrame,
    *,
    outputs: tuple[str, ...],
    excluded_features: set[str] | None = None,
    n_splits: int = 4,
    oof_seed_splits: int = DEFAULT_OOF_SEED_SPLITS,
) -> Iterator[StackedSvmFold]:
    """Create OOF LightGBM columns; warm-up rows are omitted from every fold."""
    if not outputs or len(set(outputs)) != len(outputs) or set(outputs) - set(STACKED_SVM_COLUMNS):
        raise ValueError("stacked SVM outputs must be unique, known, and nonempty")
    if not pooled_dataset.index.is_unique:
        raise ValueError("stacked SVM pooled row indices must be unique")
    if {"date", LABEL_COLUMN} - set(pooled_dataset.columns):
        raise ValueError("stacked SVM pooled rows need date and label")
    if set(STACKED_SVM_COLUMNS) & set(pooled_dataset.columns):
        raise ValueError("stacked SVM outputs must not be persisted input columns")
    outer_splits = walk_forward_splits(pooled_dataset["date"], n_splits=n_splits)
    first_train = pooled_dataset[outer_splits[0][0]]
    seed_splits = walk_forward_splits(first_train["date"], n_splits=oof_seed_splits)
    oof_blocks: list[pd.DataFrame] = []
    for seed_train_mask, seed_test_mask in seed_splits:
        _, block = _fit_and_score(
            first_train[seed_train_mask], first_train[seed_test_mask], outputs, excluded_features
        )
        oof_blocks.append(block)

    for train_mask, test_mask in outer_splits:
        train_frame = pooled_dataset[train_mask]
        test_frame = pooled_dataset[test_mask]
        historical = pd.concat(oof_blocks)
        if not historical.index.is_unique or not historical.index.isin(train_frame.index).all():
            raise ValueError("stacked SVM OOF rows must be a unique subset of training rows")
        baseline = train_frame.loc[historical.index]
        stacked_train = baseline.join(historical)
        estimators, test_columns = _fit_and_score(
            train_frame, test_frame, outputs, excluded_features
        )
        stacked_test = test_frame.join(test_columns)
        yield StackedSvmFold(baseline, stacked_train, stacked_test, estimators)
        oof_blocks.append(test_columns)
