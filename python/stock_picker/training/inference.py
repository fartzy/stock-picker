"""Builds a single lookahead-safe feature row for live "this morning" inference.

Mirrors training.dataset.build_training_frame's row construction exactly: the prior
day's full feature snapshot (computed through yesterday's close), with only
overnight_gap replaced by a value computed from today's just-observed open --
otherwise train/serve rows would be built by two different code paths that could
silently drift apart.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_picker.features.registry import DEFAULT_TTL_DAYS, check_freshness
from stock_picker.training.dataset import GAP_COLUMN
from stock_picker.training.ensemble import Ensemble, predict_ensemble

# A real overnight gap essentially never gets this large -- when the computed
# value exceeds it, a stock split or bad data between ingestion and "now" is a
# far more likely explanation than a genuine gap (see README's Known Issues).
MAX_PLAUSIBLE_GAP = 0.30


class StaleFeatureSnapshotError(Exception):
    """The prior day's feature snapshot is older than its TTL -- using it risks
    silently scoring on an incomplete/out-of-date row (Yahoo sometimes hasn't
    finalized yesterday's close yet when you pull; see README's Known Issues)."""


class ImplausibleGapError(Exception):
    """The computed overnight gap exceeds MAX_PLAUSIBLE_GAP -- almost always a
    stock split or bad data, not a real gap (see README's Known Issues)."""


def compute_overnight_gap(today_open: float, yesterday_close: float) -> float:
    """Same formula as features.candle.overnight_gap, for live scalar inputs."""
    return (today_open - yesterday_close) / yesterday_close


def build_inference_row(
    prior_day_features: pd.Series,
    today_open: float,
    yesterday_close: float,
    snapshot_date: date,
    as_of_date: date,
) -> pd.DataFrame:
    """A single-row DataFrame ready to feed to a trained ensemble's `predict_ensemble`.

    Raises `StaleFeatureSnapshotError`/`ImplausibleGapError` rather than silently
    scoring on data that's more likely wrong than right -- structural fixes for two
    of the three gotchas in README's Known Issues (the third, a same-day pull
    silently mislabeling today's in-progress row as "yesterday," is a caller
    responsibility: it depends on how the caller sources `prior_day_features`,
    which this function has no visibility into).
    """
    freshness = check_freshness(DEFAULT_TTL_DAYS, snapshot_date, as_of_date)
    if not freshness.ok:
        raise StaleFeatureSnapshotError(
            f"feature snapshot from {snapshot_date} is {freshness.age_days}d old "
            f"(ttl is {freshness.ttl_days}d) as of {as_of_date}"
        )

    gap = compute_overnight_gap(today_open, yesterday_close)
    if abs(gap) > MAX_PLAUSIBLE_GAP:
        raise ImplausibleGapError(
            f"overnight gap of {gap:.1%} exceeds the {MAX_PLAUSIBLE_GAP:.0%} plausibility "
            "bound -- likely a stock split or bad data between ingestion and now"
        )

    row = prior_day_features.copy()
    row[GAP_COLUMN] = gap
    return row.to_frame().T


def predict_signal(ensemble: Ensemble, inference_row: pd.DataFrame) -> float:
    """Predicted day-session return for a single inference row.

    Each member of `ensemble` already remembers the exact feature columns it
    was trained with (`TrainedModel.feature_names`), so there's no separate
    excluded-features set to keep in sync here.
    """
    return float(predict_ensemble(ensemble, inference_row)[0])
