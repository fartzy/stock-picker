"""Candle/gap shape features: overnight gap, body/wick sizes, close-in-range location."""

from __future__ import annotations

import pandas as pd


def overnight_gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """(today's open - yesterday's close) / yesterday's close."""
    prior_close = close.shift(1)
    return (open_ - prior_close) / prior_close


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
    return pd.DataFrame(
        {
            "overnight_gap": overnight_gap(open_, close),
            "day_range_pct": day_range_pct(high, low, close),
            "body_pct": body_pct(open_, close),
            "upper_wick_pct": upper_wick_pct(open_, high, close),
            "lower_wick_pct": lower_wick_pct(open_, low, close),
            "close_location": close_location(high, low, close),
        },
        index=history.index,
    )
