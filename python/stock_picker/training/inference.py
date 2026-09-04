"""Builds a single lookahead-safe feature row for live "this morning" inference.

Mirrors training.dataset.build_training_frame's row construction exactly: the prior
day's full feature snapshot (computed through yesterday's close), with only
overnight_gap replaced by a value computed from today's just-observed open --
otherwise train/serve rows would be built by two different code paths that could
silently drift apart.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from stock_picker.training.dataset import GAP_COLUMN
from stock_picker.training.model import feature_columns


def compute_overnight_gap(today_open: float, yesterday_close: float) -> float:
    """Same formula as features.candle.overnight_gap, for live scalar inputs."""
    return (today_open - yesterday_close) / yesterday_close


def build_inference_row(
    prior_day_features: pd.Series,
    today_open: float,
    yesterday_close: float,
) -> pd.DataFrame:
    """A single-row DataFrame ready to feed to a trained model's `.predict()`."""
    row = prior_day_features.copy()
    row[GAP_COLUMN] = compute_overnight_gap(today_open, yesterday_close)
    return row.to_frame().T


def predict_signal(
    model: lgb.Booster, inference_row: pd.DataFrame, excluded_features: set[str] | None = None
) -> float:
    """Predicted day-session return for a single inference row.

    `excluded_features` must match whatever the model was actually trained
    with (see training.main.main) -- this store only tracks current live
    state, not a per-model history, so a stale/mismatched set here would
    misalign columns against what the model expects.
    """
    columns = feature_columns(inference_row, excluded_features=excluded_features)
    return float(model.predict(inference_row[columns])[0])
