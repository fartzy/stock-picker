"""Recency-conditioned pattern features: day-session (open->close) streaks and exact
short up/down sequences, plus week-over-week same-weekday conditioning.

Same bucket-and-average mechanism as conditional_seasonality.py's setup_seasonality --
group by a categorical bucket key, average the day-session return within that bucket
-- except the conditioning average here uses a ROLLING window over the last
OCCURRENCE_WINDOW times a bucket occurred, not an expanding since-inception average.
These features are specifically about "is this less/more likely right now," not an
all-time average, so recent occurrences should be all that count and old ones should
age out entirely. A large window (e.g. 60 occurrences) would barely differ from the
expanding-since-inception average conditional_seasonality.py already provides and
defeats the point -- OCCURRENCE_WINDOW is deliberately small.

Bucket sparsity is a bigger concern here than in conditional_seasonality.py: a 3-day
up/down sequence is already 2**3 = 8 buckets (no dead-zone/flat state -- day-session
returns are practically never exactly zero, unlike the overnight gap conditional_
seasonality.py bucket, so a flat state isn't meaningful the same way). Combined with
a rolling window of only OCCURRENCE_WINDOW occurrences, expect these seasonality
columns to be sparse/NaN for a while even on mature tickers -- that's expected, not a
bug; see coverage_report in catalog.py.

Per-ticker only for this first pass, same rollout order as setup_seasonality shipping
before pooled_setup_seasonality. All three seasonality columns are lookahead-safe by
the same discipline as setup_seasonality: the rolling mean is inclusive of the
current row's own same-day return, so on its own this is NOT safe to use as a
same-day feature -- it only becomes safe via training/dataset.py's blanket
features.shift(1).
"""

from __future__ import annotations

import pandas as pd

from stock_picker.features.conditional_seasonality import (
    PRIOR_RETURN_FLAT_THRESHOLD,
    bucket_signal,
)

SEQUENCE_WINDOW = 3
OCCURRENCE_WINDOW = 10
STREAK_CAP = 3
WEEKDAY_LAG_DAYS = 5


def day_session_return(history: pd.DataFrame) -> pd.Series:
    """(Close - Open) / Open -- the day's own open->close move. Same formula as
    candle.body_pct and training/dataset.py's label; computed locally here rather
    than imported so this module has no dependency on candle.py, matching how
    conditional_seasonality.py computes gap/prior-return locally instead of
    importing candle.overnight_gap."""
    return (history["Close"] - history["Open"]) / history["Open"]


def day_session_streak(returns: pd.Series, cap: int = STREAK_CAP) -> pd.Series:
    """Signed, capped count of consecutive up/down day-session days: +2 = two
    straight day-session gains, -3 (capped) = three-or-more straight day-session
    losses. Same loop-based approach as momentum.consecutive_day_streak (inherently
    sequential state, not a plain rolling window) but keyed off the day-session
    (open->close) direction instead of close-to-close, and capped so the bucket used
    by the seasonality conditioning below doesn't fragment past what a handful of
    occurrences can support.
    """
    streak = pd.Series(0.0, index=returns.index)
    current = 0.0
    for i, r in enumerate(returns.to_numpy()):
        if pd.isna(r):
            current = 0.0
        elif r >= 0:
            current = min(current + 1, cap) if current > 0 else 1.0
        else:
            current = max(current - 1, -cap) if current < 0 else -1.0
        streak.iloc[i] = current
    return streak


def sequence_bucket(returns: pd.Series, window: int = SEQUENCE_WINDOW) -> pd.Series:
    """Exact up/down sequence of the trailing `window` day-session returns as a
    string key, e.g. "UDU" -- encodes order, not just streak length/direction (a
    "UDU" zigzag and a "DUD" reversed zigzag are different buckets despite both
    having zero net streak). NaN until `window` consecutive non-NaN days are
    available.
    """
    label = pd.Series(pd.NA, index=returns.index, dtype="object")
    valid = returns.notna()
    label[valid & (returns >= 0)] = "U"
    label[valid & (returns < 0)] = "D"

    keys = pd.Series(pd.NA, index=returns.index, dtype="object")
    values = label.to_numpy()
    for i in range(window - 1, len(values)):
        trailing = values[i - window + 1 : i + 1]
        if pd.isna(trailing).any():
            continue
        keys.iloc[i] = "".join(trailing)
    return keys


def weekday_lag_return(returns: pd.Series, lag: int = WEEKDAY_LAG_DAYS) -> pd.Series:
    """Day-session return `lag` trading days ago -- default 5, the same weekday one
    calendar week prior (today's Thursday vs. last Thursday). Exposed both as a raw
    numeric feature (dense, no bucketing needed) and as the input to
    weekday_lag_seasonality's bucket below.
    """
    return returns.shift(lag)


def weekday_lag_bucket(history: pd.DataFrame, lag_return: pd.Series) -> pd.Series:
    """Combine day-of-week with the discretized direction/size of `lag_return` (last
    week's same-weekday day-session return) into a single bucket key, e.g. "3_1" =
    Thursday after a big up move last Thursday. Same "zip over the valid mask, NaN
    propagates rather than landing in a spurious bucket" idiom as
    conditional_seasonality.setup_bucket.
    """
    weekday = pd.Series(history.index.dayofweek, index=history.index)
    lag_bucket = bucket_signal(lag_return, PRIOR_RETURN_FLAT_THRESHOLD)

    key = pd.Series(pd.NA, index=history.index, dtype="object")
    valid = lag_bucket.notna()
    key[valid] = [f"{d}_{b}" for d, b in zip(weekday[valid], lag_bucket[valid])]
    return key


def rolling_seasonality(
    returns: pd.Series, bucket: pd.Series, window: int = OCCURRENCE_WINDOW
) -> pd.Series:
    """Shared conditioning mechanism for every pattern feature in this module:
    average day-session return over only the last `window` PRIOR OCCURRENCES of this
    exact bucket -- not calendar days, and not expanding-since-inception like
    conditional_seasonality.setup_seasonality. Old occurrences age out entirely past
    `window` occurrences, so the average tracks the ticker's current regime instead
    of its all-time history.
    """
    return returns.groupby(bucket).transform(lambda s: s.rolling(window, min_periods=1).mean())


def build_pattern_features(history: pd.DataFrame) -> pd.DataFrame:
    returns = day_session_return(history)
    streak = day_session_streak(returns)
    sequence = sequence_bucket(returns)
    lag_return = weekday_lag_return(returns)
    weekday_bucket = weekday_lag_bucket(history, lag_return)

    return pd.DataFrame(
        {
            "day_session_streak": streak,
            "day_session_streak_seasonality": rolling_seasonality(returns, streak),
            "pattern_sequence_seasonality_3d": rolling_seasonality(returns, sequence),
            "weekday_lag_return": lag_return,
            "weekday_lag_seasonality": rolling_seasonality(returns, weekday_bucket),
        },
        index=history.index,
    )
