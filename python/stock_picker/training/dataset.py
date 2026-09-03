"""Builds lookahead-safe, labeled training rows from raw prices and computed features.

The label for day t is the same-day open->close ("day session") return. Predicting it
using day t's own feature row would leak the label -- most feature columns are computed
from day t's full OHLCV, including Close_t, which is exactly what we're predicting. The
fix: use features computed through day t-1 (features.shift(1)), except overnight_gap,
which is legitimately knowable at day t's open and must NOT be shifted.
"""

from __future__ import annotations

import pandas as pd

LABEL_COLUMN = "label_day_session_return"
GAP_COLUMN = "overnight_gap"


def day_session_return(history: pd.DataFrame) -> pd.Series:
    """(Close_t - Open_t) / Open_t for every day."""
    return (history["Close"] - history["Open"]) / history["Open"]


def build_training_frame(history: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per date with a lookahead-safe feature vector and the day-session label.

    Drops the first row (no prior day to shift features from) and any row with an
    undefined label. Remaining NaNs (from long-window features without enough trailing
    history) are left as-is -- LightGBM splits on missing values natively.
    """
    label = day_session_return(history)

    shifted = features.shift(1)
    shifted[GAP_COLUMN] = features[GAP_COLUMN]
    shifted[LABEL_COLUMN] = label

    frame = shifted.iloc[1:]
    return frame[frame[LABEL_COLUMN].notna()]


def build_pooled_dataset(
    histories: dict[str, pd.DataFrame],
    features_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate build_training_frame's output across every ticker.

    Adds "ticker" and "date" metadata columns for splitting/tracing -- these are not
    model features (see training.model.FEATURE_COLUMNS).
    """
    frames = []
    for ticker, history in histories.items():
        frame = build_training_frame(history, features_by_ticker[ticker])
        frame = frame.reset_index(names="date")
        frame.insert(0, "ticker", ticker)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
