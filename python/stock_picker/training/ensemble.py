"""Combines several `TrainedModel`s (possibly different types, possibly each
trained on a different feature subset) into one weighted-average predictor --
so no single model family's blind spots decide the final signal alone.

A single-model run is just a one-member ensemble with weight 1.0; every other
module (storage, importance, inference) only ever sees an `Ensemble`, so
there's one code path instead of two parallel single-vs-ensemble cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN, score_direction_margin
from stock_picker.training.model import (
    RANK_MODEL_TYPE,
    EvaluationMetrics,
    TrainedModel,
    feature_columns,
    predict,
    train_model,
)


@dataclass
class ModelSpec:
    model_type: str
    params: dict | None = None
    excluded_features: set[str] | None = None
    included_features: set[str] | None = None
    weight: float = 1.0


@dataclass
class Ensemble:
    members: list[TrainedModel]
    weights: list[float]
    # Kept in the same archive as the LightGBM member that consumes its
    # margin. Old pickles do not have this field; scoring uses getattr.
    direction_svc: TrainedModel | None = None


def train_ensemble(train_frame: pd.DataFrame, specs: list[ModelSpec]) -> Ensemble:
    for spec in specs:
        if (
            spec.model_type != "lightgbm"
            and DIRECTION_MARGIN_COLUMN in feature_columns(
                train_frame, spec.excluded_features, spec.included_features
            )
        ):
            raise ValueError(
                f"{DIRECTION_MARGIN_COLUMN} is only supported for LightGBM members; "
                f"{spec.model_type} would consume it"
            )
    members = [
        train_model(
            spec.model_type,
            train_frame,
            params=spec.params,
            excluded_features=spec.excluded_features,
            included_features=spec.included_features,
        )
        for spec in specs
    ]
    return Ensemble(members=members, weights=[spec.weight for spec in specs])


def predict_ensemble(ensemble: Ensemble, frame: pd.DataFrame) -> np.ndarray:
    unsupported = [
        member.model_type
        for member in ensemble.members
        if member.model_type != "lightgbm" and DIRECTION_MARGIN_COLUMN in member.feature_names
    ]
    if unsupported:
        raise ValueError(
            f"{DIRECTION_MARGIN_COLUMN} is only supported for LightGBM members; "
            f"found {unsupported}"
        )
    if any(
        member.model_type == "lightgbm" and DIRECTION_MARGIN_COLUMN in member.feature_names
        for member in ensemble.members
    ):
        direction_svc = getattr(ensemble, "direction_svc", None)
        if direction_svc is None:
            raise ValueError("LightGBM expects direction margin but its fitted SVC is missing")
        # Ignore a supplied margin: it could be stale, spoofed, or generated
        # by a different SVC than the one paired with this archived model.
        frame = frame.assign(**{
            DIRECTION_MARGIN_COLUMN: score_direction_margin(frame, direction_svc)
        })
    total_weight = sum(ensemble.weights)
    blended = sum(
        predict(member, frame) * weight for member, weight in zip(ensemble.members, ensemble.weights)
    )
    return np.asarray(blended) / total_weight


def evaluate_ensemble(ensemble: Ensemble, test_frame: pd.DataFrame) -> EvaluationMetrics:
    predictions = predict_ensemble(ensemble, test_frame)
    actual = test_frame[LABEL_COLUMN].to_numpy()

    return EvaluationMetrics(
        mae=float(np.mean(np.abs(predictions - actual))),
        directional_accuracy=float(np.mean(np.sign(predictions) == np.sign(actual))),
        n_test_rows=len(test_frame),
    )


@dataclass
class EnsembleMemberInfo:
    model_type: str
    weight: float
    feature_count: int


def ensemble_composition(ensemble: Ensemble) -> list[EnsembleMemberInfo]:
    """Which model types make up this ensemble, their blend weights, and how
    many features each was trained on -- the "what's actually in here" view
    that `/api/feature-importance` doesn't answer on its own (it shows each
    feature's *impact*, not the ensemble's *composition*)."""
    return [
        EnsembleMemberInfo(model_type=member.model_type, weight=weight, feature_count=len(member.feature_names))
        for member, weight in zip(ensemble.members, ensemble.weights)
    ]


def ensemble_feature_names(ensemble: Ensemble) -> frozenset[str]:
    """Union of every member's own persisted `feature_names` -- what this
    specific loaded model (not the current global pruned-features state,
    which may have changed since it was trained) actually reads. The one
    safe source for "which open-known columns can live-row building skip
    computing for this model" (see `live_rows.prepare_live_rows`):
    `predict()` reindexes to a model's own `feature_names`, so a column
    missing from a DIFFERENT (older or newer) model's list would silently
    reindex to NaN instead of its real value if this were derived from
    anything other than the model actually about to score the row.
    """
    names = frozenset().union(*(set(member.feature_names) for member in ensemble.members))
    direction_svc = getattr(ensemble, "direction_svc", None)
    if direction_svc is not None:
        names |= frozenset(direction_svc.feature_names)
    return names


def selected_model_specs() -> list[ModelSpec] | None:
    """Wiring: reads the persisted composable model-type choices (see
    `storage/training_config_store.py`), or None if nothing's been chosen
    yet -- meaning "use training/main.py's own default composition"."""
    choices = TrainingConfigStore().read().model_choices
    if choices is None:
        return None
    return [ModelSpec(choice.model_type, weight=choice.weight) for choice in choices]


def partition_model_specs(specs: list[ModelSpec] | None) -> tuple[list[ModelSpec] | None, bool]:
    """Split UI/CLI specs into return-ensemble members vs the rank model.

    Rank scores are not percents -- they cannot be weight-averaged with
    LightGBM/Ridge. None specs = default return composition *and* train rank.
    Rank-only choices still train the default return ensemble so the 0.5%
    list is never left empty.
    """
    if specs is None:
        return None, True
    predictive = [spec for spec in specs if spec.model_type != RANK_MODEL_TYPE]
    wants_rank = any(spec.model_type == RANK_MODEL_TYPE for spec in specs)
    return (predictive or None, wants_rank)
