"""Strictly prior evidence for morning U/D sequences; shared research/live builder.

These candidate inputs are not enabled in the production feature pipeline yet.
Counts, dispersion and recency accompany the existing two/three/four-day patterns.
All constants are fixed before evaluation; no outcome selects a window or prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_picker.features.conditional_seasonality import GAP_FLAT_THRESHOLD, bucket_signal
from stock_picker.features.dip_context import complete_dip_history
from stock_picker.features.open_pattern_seasonality import combine_keys, completed_sequence
from stock_picker.storage.price_store import sanitize_price_history

PRIOR_OBSERVATIONS = 20.0
HALF_LIFE_SESSIONS = 63.0
SEQUENCE_LENGTHS = (2, 3, 4)
STATISTICS = ("count", "std", "age", "shrunk_mean", "recent_mean")
PATTERN_EVIDENCE_COLUMNS = tuple(
    f"seq{length}_open3_{stat}" for length in SEQUENCE_LENGTHS for stat in STATISTICS
)


def prior_pattern_evidence(values: pd.Series, buckets: pd.Series) -> pd.DataFrame:
    """Emit before updating: neither today's outcome nor a future row is visible.

Count means valid earlier outcomes for this exact ticker/pattern. Age and decay
use session positions, including inserted missing exchange sessions. Shrinkage
uses this ticker's strictly prior unconditional mean, with 20 pseudo-observations.
The recent mean has a 63-session half-life. Unknown buckets remain wholly unknown;
a known but unseen bucket has count zero and no dispersion/age/recent mean.
"""
    if not values.index.equals(buckets.index):
        raise ValueError("pattern values and buckets must have the same index")
    result = np.full((len(values), len(STATISTICS)), np.nan)
    # count, mean, centered sum of squares, last observation, decayed sum, weight
    states: dict[str, list[float]] = {}
    total, total_count = 0.0, 0
    for pos, (value, bucket) in enumerate(zip(values, buckets)):
        known = pd.notna(bucket)
        if known:
            key = str(bucket)
            state = states.get(key, [0.0, 0.0, 0.0, float(pos), 0.0, 0.0])
            count, mean, m2, last, weighted_sum, weight = state
            decay = 2.0 ** (-(pos - last) / HALF_LIFE_SESSIONS)
            weighted_sum, weight = weighted_sum * decay, weight * decay
            result[pos, 0] = count
            if count > 1:
                result[pos, 1] = np.sqrt(max(0.0, m2) / (count - 1))
            if count:
                result[pos, 2] = pos - last
            if total_count:
                result[pos, 3] = (
                    count * mean + PRIOR_OBSERVATIONS * total / total_count
                ) / (count + PRIOR_OBSERVATIONS)
            if weight > 0:
                result[pos, 4] = weighted_sum / weight
            if pd.notna(value) and np.isfinite(value):
                difference = value - mean
                new_mean = mean + difference / (count + 1)
                states[key] = [count + 1, new_mean, m2 + difference * (value - new_mean),
                               float(pos), weighted_sum + value, weight + 1]
        if pd.notna(value) and np.isfinite(value):
            total += value
            total_count += 1
    return pd.DataFrame(result, index=values.index, columns=STATISTICS)


def build_pattern_evidence(history: pd.DataFrame) -> pd.DataFrame:
    """Completed history plus today's opening price; same formula in replay/live."""
    if history.empty:
        return pd.DataFrame(index=history.index, columns=PATTERN_EVIDENCE_COLUMNS, dtype=float)
    expanded = complete_dip_history(sanitize_price_history(history))
    session = expanded["Close"].div(expanded["Open"]).sub(1)
    gap = expanded["Open"].div(expanded["Close"].shift(1)).sub(1)
    opening = bucket_signal(gap, GAP_FLAT_THRESHOLD)
    families = []
    for length in SEQUENCE_LENGTHS:
        buckets = combine_keys(completed_sequence(session, length), opening)
        evidence = prior_pattern_evidence(session, buckets)
        families.append(evidence.add_prefix(f"seq{length}_open3_"))
    return pd.concat(families, axis=1).reindex(history.index)


def pattern_evidence_row(history: pd.DataFrame, today_open: float, session_date: str) -> pd.Series:
    """Score an explicit session date; callers supply completed sessions only."""
    today = pd.Timestamp(session_date)
    if history.empty or not np.isfinite(today_open) or today_open <= 0:
        return pd.Series(np.nan, index=PATTERN_EVIDENCE_COLUMNS)
    if today <= history.index[-1] or today != today.normalize():
        raise ValueError("entry session must be normalized and after completed history")
    row = pd.DataFrame(np.nan, index=pd.DatetimeIndex([today]), columns=history.columns)
    row.loc[today, "Open"] = today_open
    return build_pattern_evidence(pd.concat([history, row])).iloc[-1]
