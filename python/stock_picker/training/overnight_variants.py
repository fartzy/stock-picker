"""Named same-day prediction inputs for the Rank-cohort next-open model.

Each variant predicts the open-to-close return. Its training frame is a full
cross-section of earlier sessions, never just the selected Rank names. The
Rank filter is applied only after these morning predictions are generated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.ensemble import Ensemble, ModelSpec, train_ensemble, predict_ensemble
from stock_picker.training.model import feature_columns


@dataclass(frozen=True)
class MorningVariant:
    column: str
    model_type: str
    window_sessions: int | None


MORNING_VARIANTS = (
    MorningVariant("day_fit_lgbm_20_session_return", "lightgbm", 20),
    MorningVariant("day_fit_lgbm_120_session_return", "lightgbm", 120),
    MorningVariant("day_fit_ridge_return", "ridge", None),
)
VARIANT_OUTPUT_COLUMNS = tuple(variant.column for variant in MORNING_VARIANTS)


def variant_feature_names(frame: pd.DataFrame, fit_specs: list[ModelSpec]) -> frozenset[str]:
    """Use the selected Fit's raw inputs, excluding already-stacked SVM scores."""
    if not fit_specs:
        raise ValueError("at least one selected Fit member is required for morning variants")
    names = set().union(*(
        set(feature_columns(frame, spec.excluded_features, spec.included_features))
        for spec in fit_specs
    )) - set(STACKED_SVM_COLUMNS)
    if not names:
        raise ValueError("selected Fit has no raw features for morning variants")
    return frozenset(names)


def train_morning_variants(
    day_training_frame: pd.DataFrame,
    *,
    included_features: frozenset[str],
    scoring_date: pd.Timestamp | None = None,
) -> dict[str, Ensemble]:
    """Fit named variants on earlier full-universe sessions only.

    ``scoring_date`` is mandatory in historical folds. The final serving fit
    omits it and is guarded by the archived ``day_model_trained_through``.
    """
    dates = pd.to_datetime(day_training_frame["date"], errors="raise")
    if day_training_frame.empty or dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("morning variant training needs dated, timezone-naive rows")
    if scoring_date is not None and dates.max() >= pd.Timestamp(scoring_date):
        raise ValueError("morning variants must be fitted before the scoring session")
    missing = included_features - set(day_training_frame)
    if missing:
        raise ValueError(f"morning variant inputs are missing: {sorted(missing)}")
    ordered_sessions = dates.drop_duplicates().sort_values()
    models = {}
    for variant in MORNING_VARIANTS:
        selected_dates = (
            ordered_sessions.iloc[-variant.window_sessions:]
            if variant.window_sessions is not None else ordered_sessions
        )
        train = day_training_frame.loc[dates.isin(selected_dates)]
        models[variant.column] = train_ensemble(
            train,
            [ModelSpec(variant.model_type, included_features=set(included_features))],
        )
    return models


def score_morning_variants(
    open_known_rows: pd.DataFrame, models: dict[str, Ensemble],
) -> pd.DataFrame:
    """Score exactly the named estimators embedded in the overnight artifact."""
    if set(models) != set(VARIANT_OUTPUT_COLUMNS) or any(
        models[column] is None for column in VARIANT_OUTPUT_COLUMNS
    ):
        raise ValueError("overnight artifact lacks its complete named morning variants")
    scores = pd.DataFrame({
        variant.column: predict_ensemble(models[variant.column], open_known_rows)
        for variant in MORNING_VARIANTS
    }, index=open_known_rows.index)
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("morning variant predictions must be finite")
    return scores
