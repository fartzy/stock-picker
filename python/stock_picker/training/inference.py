"""Builds a single lookahead-safe feature row for live "this morning" inference.

Mirrors training.dataset.build_training_frame's row construction: the prior
day's full feature snapshot (computed through yesterday's close), with the
open-known columns (overnight_gap plus recency-pattern seasonality) replaced
by values computed from today's just-observed open -- otherwise train/serve
rows would be built by two different code paths that could silently drift
apart.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_picker.features.open_pattern_seasonality import (
    OPEN_KNOWN_COLUMNS,
    open_known_feature_row,
)
from stock_picker.features.registry import DEFAULT_TTL_DAYS, check_freshness
from stock_picker.training.dataset import GAP_COLUMN
from stock_picker.training.ensemble import Ensemble, predict_ensemble


class StaleFeatureSnapshotError(Exception):
    """The prior day's feature snapshot is older than its TTL -- using it risks
    silently scoring on an incomplete/out-of-date row (Yahoo sometimes hasn't
    finalized yesterday's close yet when you pull; see README's Known Issues)."""


def compute_overnight_gap(today_open: float, yesterday_close: float) -> float:
    """Same formula as features.candle.overnight_gap, for live scalar inputs."""
    return (today_open - yesterday_close) / yesterday_close


def build_inference_row(
    prior_day_features: pd.Series,
    today_open: float,
    yesterday_close: float,
    snapshot_date: date,
    as_of_date: date,
    prior_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """A single-row DataFrame ready to feed to a trained ensemble's `predict_ensemble`.

    Raises `StaleFeatureSnapshotError` rather than silently scoring on a
    feature snapshot that's more likely stale than right. A large overnight
    gap is a real open, not a reason to refuse the row -- the quote fetcher
    already required the print to be dated today.
    """
    freshness = check_freshness(DEFAULT_TTL_DAYS, snapshot_date, as_of_date)
    if not freshness.ok:
        raise StaleFeatureSnapshotError(
            f"feature snapshot from {snapshot_date} is {freshness.age_days}d old "
            f"(ttl is {freshness.ttl_days}d) as of {as_of_date}"
        )

    gap = compute_overnight_gap(today_open, yesterday_close)
    row = prior_day_features.copy()
    row[GAP_COLUMN] = gap
    if prior_history is not None and not prior_history.empty:
        open_known = open_known_feature_row(prior_history, today_open)
        for column in OPEN_KNOWN_COLUMNS:
            if column in row.index and column in open_known.index:
                row[column] = open_known[column]
    return row.to_frame().T


def predict_signal(ensemble: Ensemble, inference_row: pd.DataFrame) -> float:
    """Predicted day-session return for a single inference row.

    Each member of `ensemble` already remembers the exact feature columns it
    was trained with (`TrainedModel.feature_names`), so there's no separate
    excluded-features set to keep in sync here.
    """
    return float(predict_ensemble(ensemble, inference_row)[0])
