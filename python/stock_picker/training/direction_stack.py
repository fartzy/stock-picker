"""Chronological direction-SVC margins for Fit's LightGBM training rows.

The first outer training period has no preceding outer test block. Three
smaller walk-forward fits score its later dates; the initial seed period is
warm-up and is omitted from both baseline and stacked LightGBM training.
Each later outer test block supplies historical OOF values to the next fold.
The SVC itself always fits on the full, unstacked prior training period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import TrainedModel, decision_scores, train_svc_direction
from stock_picker.training.splits import walk_forward_splits

DIRECTION_MARGIN_COLUMN = "svc_direction_margin"
DEFAULT_OUTER_SPLITS = 4
DEFAULT_OOF_SEED_SPLITS = 3


@dataclass(frozen=True)
class DirectionMarginFold:
    """Aligned frames for one fold and the SVC used to score its test rows."""

    fold: int
    baseline_train: pd.DataFrame
    stacked_train: pd.DataFrame
    stacked_test: pd.DataFrame
    direction_svc: TrainedModel


def score_direction_margin(frame: pd.DataFrame, direction_svc: TrainedModel) -> pd.Series:
    """Score a frame in row order, rejecting missing inputs and invalid margins.

    The caller is responsible for ensuring the estimator was fitted on
    permissible earlier dates; ``_fit_and_score`` enforces that for OOF rows.
    This scorer can also be reused on an open-known inference frame.
    """
    if direction_svc.model_type != "svc_direction":
        raise ValueError("direction margin requires a fitted direction SVC")
    missing = set(direction_svc.feature_names) - set(frame.columns)
    if missing:
        raise ValueError(f"direction SVC inputs are missing: {sorted(missing)}")
    if DIRECTION_MARGIN_COLUMN in direction_svc.feature_names:
        raise ValueError("direction SVC cannot use its own margin as an input")
    # The saved median imputer permits ordinary partial gaps, but an entirely
    # missing source vector can otherwise yield a plausible constant margin.
    if frame[direction_svc.feature_names].isna().all(axis=1).any():
        raise ValueError("direction SVC inputs are all missing for at least one row")
    values = np.asarray(decision_scores(direction_svc, frame), dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("direction SVC margins must be finite and row-aligned")
    return pd.Series(values, index=frame.index, name=DIRECTION_MARGIN_COLUMN)


def _fit_and_score(
    fit_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    excluded_features: set[str] | None,
    included_features: set[str] | None,
) -> tuple[TrainedModel, pd.Series]:
    """Fit only on dates strictly preceding every scored date."""
    if fit_frame.empty or score_frame.empty or fit_frame["date"].max() >= score_frame["date"].min():
        raise ValueError("direction SVC requires strictly earlier fitting dates")
    direction_svc = train_svc_direction(
        fit_frame,
        excluded_features=excluded_features,
        included_features=included_features,
    )
    return direction_svc, score_direction_margin(score_frame, direction_svc)


def iter_direction_margin_folds(
    pooled_dataset: pd.DataFrame,
    *,
    excluded_features: set[str] | None = None,
    included_features: set[str] | None = None,
    n_splits: int = DEFAULT_OUTER_SPLITS,
    oof_seed_splits: int = DEFAULT_OOF_SEED_SPLITS,
) -> Iterator[DirectionMarginFold]:
    """Yield four chronological LightGBM views and their fitted direction SVCs.

    ``pooled_dataset`` must already exclude held-out tickers. Its original
    unique row index is preserved across all outputs. Only rows with an
    earlier-date SVC prediction enter ``baseline_train`` and ``stacked_train``;
    both have identical identities, labels, and ordinary features.
    """
    if n_splits < 1 or oof_seed_splits < 1:
        raise ValueError("direction SVC fold counts must be positive")
    if not pooled_dataset.index.is_unique:
        raise ValueError("direction SVC pooled row indices must be unique")
    required = {"date", LABEL_COLUMN}
    missing = required - set(pooled_dataset.columns)
    if missing:
        raise ValueError(f"direction SVC pooled columns are missing: {sorted(missing)}")
    if DIRECTION_MARGIN_COLUMN in pooled_dataset.columns:
        raise ValueError("direction SVC margin must not be a persisted input column")

    outer_splits = walk_forward_splits(pooled_dataset["date"], n_splits=n_splits)
    first_train = pooled_dataset[outer_splits[0][0]]
    seed_splits = walk_forward_splits(first_train["date"], n_splits=oof_seed_splits)
    oof_blocks: list[pd.Series] = []
    for seed_train_mask, seed_test_mask in seed_splits:
        _, margin = _fit_and_score(
            first_train[seed_train_mask],
            first_train[seed_test_mask],
            excluded_features,
            included_features,
        )
        oof_blocks.append(margin)

    for fold, (train_mask, test_mask) in enumerate(outer_splits, start=1):
        train_frame = pooled_dataset[train_mask]
        test_frame = pooled_dataset[test_mask]
        historical_margins = pd.concat(oof_blocks)
        if (not historical_margins.index.is_unique
                or not historical_margins.index.isin(train_frame.index).all()):
            raise ValueError("direction SVC OOF rows must be a unique subset of training rows")
        baseline_train = train_frame.loc[historical_margins.index]
        stacked_train = baseline_train.assign(**{DIRECTION_MARGIN_COLUMN: historical_margins})
        direction_svc, test_margin = _fit_and_score(
            train_frame, test_frame, excluded_features, included_features
        )
        stacked_test = test_frame.assign(**{DIRECTION_MARGIN_COLUMN: test_margin})
        yield DirectionMarginFold(
            fold=fold,
            baseline_train=baseline_train,
            stacked_train=stacked_train,
            stacked_test=stacked_test,
            direction_svc=direction_svc,
        )
        oof_blocks.append(test_margin)
