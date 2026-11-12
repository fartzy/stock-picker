"""Builds lookahead-safe, labeled training rows from raw prices and computed features.

The label for day t is the same-day open->close ("day session") return. Predicting it
using day t's own feature row would leak the label -- most feature columns are computed
from day t's full OHLCV, including Close_t, which is exactly what we're predicting. The
fix: use features computed through day t-1 (features.shift(1)), except the
open-known columns (including overnight_gap, recency-pattern seasonality and
the open-dependent dip fields), which are knowable at day t's open and must NOT be
shifted. Those columns never use Close_t -- they key off completed days
plus Open_t -- so leaving them unshifted is not a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_picker.features.dip_context import DIP_CLOSE_KNOWN_COLUMNS, DIP_OPEN_KNOWN_COLUMNS, complete_dip_history
from stock_picker.features.open_pattern_seasonality import OPEN_KNOWN_COLUMNS
from stock_picker.features.structure import CLUSTER_GAP_COLUMN
from stock_picker.features.weather import WEATHER_COLUMNS
from stock_picker.features.kronos import CONTRACT_ATTR, KRONOS_COLUMNS, frame_contract

LABEL_COLUMN = "label_day_session_return"
GAP_COLUMN = "overnight_gap"
OPEN_KNOWN_FEATURE_COLUMNS = frozenset(
    {GAP_COLUMN, "spy_overnight_gap", CLUSTER_GAP_COLUMN, *OPEN_KNOWN_COLUMNS,
     *DIP_OPEN_KNOWN_COLUMNS, *WEATHER_COLUMNS, *KRONOS_COLUMNS}
)


def day_session_return(history: pd.DataFrame) -> pd.Series:
    """(Close_t - Open_t) / Open_t for every day."""
    open_price = history["Open"].where(history["Open"].gt(0))
    return (history["Close"] - open_price) / open_price


def build_training_frame(history: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per date with a lookahead-safe feature vector and the day-session label.

    Drops the first row (no prior day to shift features from) and any row with an
    undefined label. Remaining NaNs (from long-window features without enough trailing
    history) are left as-is -- LightGBM splits on missing values natively.
    """
    label = day_session_return(history)

    shifted = features.shift(1)
    present = [column for column in OPEN_KNOWN_FEATURE_COLUMNS if column in features.columns]
    shifted[present] = features[present]
    dip_close = [column for column in DIP_CLOSE_KNOWN_COLUMNS if column in features.columns]
    if dip_close:
        calendar_index = complete_dip_history(history).index
        dip_snapshots = features[dip_close].reindex(calendar_index)
        missing_sessions = calendar_index.difference(history.index)
        # A missing completed bar makes every close-known dip measurement
        # unknown; the explicit coverage flag is known to be false.
        if "dip_low_history_valid" in dip_close:
            dip_snapshots.loc[missing_sessions, "dip_low_history_valid"] = 0.0
        shifted[dip_close] = dip_snapshots.shift(1).reindex(features.index)
    shifted[LABEL_COLUMN] = label

    frame = shifted.iloc[1:]
    return frame[np.isfinite(frame[LABEL_COLUMN])]


def build_pooled_dataset(
    histories: dict[str, pd.DataFrame],
    features_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate build_training_frame's output across every ticker.

    Adds "ticker" and "date" metadata columns for splitting/tracing -- these are not
    model features (see training.model.FEATURE_COLUMNS).
    """
    frames = []
    contracts = []
    for ticker, history in histories.items():
        contracts.append(frame_contract(features_by_ticker[ticker], features_by_ticker[ticker].columns))
        frame = build_training_frame(history, features_by_ticker[ticker])
        frame = frame.reset_index(names="date")
        frame.insert(0, "ticker", ticker)
        frames.append(frame)

    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("pooled features mix Kronos contracts or missing provenance; rebuild features")
    pooled = pd.concat(frames, ignore_index=True)
    if contracts and contracts[0] is not None:
        pooled.attrs[CONTRACT_ATTR] = contracts[0]
    return pooled
