"""Candle/gap shape features: overnight gap, body/wick sizes, close-in-range location."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_picker.features.volume import volume_zscore


def overnight_gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """(today's open - yesterday's close) / yesterday's close."""
    prior_close = close.shift(1)
    return (open_ - prior_close) / prior_close


def gap_volume_interaction(gap: pd.Series, volume: pd.Series) -> pd.Series:
    """Overnight gap scaled by how unusual today's volume is (volume_zscore) --
    a gap on unusually high volume ("confirmed" by real participation) is a
    different signal than the same-size gap on unusually low volume, but a
    plain overnight_gap column can't distinguish them. Research on
    overnight-to-intraday reversal (e.g. Della Corte & Kosowski) motivates
    treating the overnight window as an unusually information-dense predictor
    of the next session's move -- this is a first cut at letting the model
    use gap size and volume-confirmation jointly instead of as two
    independent, equally-weighted columns among ~30-100 others.
    """
    return gap * volume_zscore(volume)


def gap_streak(gap: pd.Series) -> pd.Series:
    """Signed count of consecutive same-direction overnight gaps: +3 = 3
    straight gap-up days, -2 = 2 straight gap-down days, 0 on a flat/no-gap
    day. Same loop-based approach as momentum.consecutive_day_streak
    (inherently sequential state, not a plain rolling window) -- a
    gap-direction analog to that feature, since gap persistence (is the
    market repeatedly surprised the same way) is a distinct question from
    plain close-to-close streaks.
    """
    direction = np.sign(gap)
    streak = pd.Series(0.0, index=gap.index)
    current = 0.0
    for i, d in enumerate(direction.to_numpy()):
        if pd.isna(d) or d == 0:
            current = 0.0
        elif d > 0:
            current = current + 1 if current > 0 else 1.0
        else:
            current = current - 1 if current < 0 else -1.0
        streak.iloc[i] = current
    return streak


def day_range_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high - low) / close


def body_pct(open_: pd.Series, close: pd.Series) -> pd.Series:
    return (close - open_) / open_


def upper_wick_pct(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    return (high - body_top) / open_


def lower_wick_pct(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    return (body_bottom - low) / open_


def close_location(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """0 = closed at the day's low, 1 = closed at the day's high."""
    return (close - low) / (high - low)


def build_candle_features(history: pd.DataFrame) -> pd.DataFrame:
    open_ = history["Open"]
    high = history["High"]
    low = history["Low"]
    close = history["Close"]
    volume = history["Volume"]
    gap = overnight_gap(open_, close)
    return pd.DataFrame(
        {
            "overnight_gap": gap,
            "gap_volume_interaction": gap_volume_interaction(gap, volume),
            "gap_streak": gap_streak(gap),
            "day_range_pct": day_range_pct(high, low, close),
            "body_pct": body_pct(open_, close),
            "upper_wick_pct": upper_wick_pct(open_, high, close),
            "lower_wick_pct": lower_wick_pct(open_, low, close),
            "close_location": close_location(high, low, close),
        },
        index=history.index,
    )
