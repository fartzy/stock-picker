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

from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import EvaluationMetrics, TrainedModel, predict, train_model


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


def train_ensemble(train_frame: pd.DataFrame, specs: list[ModelSpec]) -> Ensemble:
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
