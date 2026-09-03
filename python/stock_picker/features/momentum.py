"""Momentum/return features: simple and log returns over multiple windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [1, 3, 5, 10, 20, 60, 120]
SPREAD_WINDOW_PAIRS = [(5, 20), (10, 60), (20, 120)]


def simple_return(close: pd.Series, window: int) -> pd.Series:
    """Percent change over `window` trailing trading days."""
    return close.pct_change(window)


def log_return(close: pd.Series, window: int) -> pd.Series:
    """Log return over `window` trailing trading days."""
    return np.log(close / close.shift(window))


def momentum_spread(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    """Short-horizon return minus long-horizon return -- is the trend accelerating
    (positive) or decelerating (negative) relative to its own longer-term pace?"""
    return simple_return(close, short_window) - simple_return(close, long_window)


def consecutive_day_streak(close: pd.Series) -> pd.Series:
    """Signed count of consecutive up/down closes: +3 = 3 straight up days,
    -2 = 2 straight down days, 0 on a flat (no-change) day.

    Inherently sequential state, not a plain rolling window -- a loop is the
    clearest way to express it at this data scale (hundreds of rows per ticker).
    """
    direction = np.sign(close.diff())
    streak = pd.Series(0.0, index=close.index)
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


def build_momentum_features(history: pd.DataFrame) -> pd.DataFrame:
    """Return simple_return_{n}d and log_return_{n}d for each window in WINDOWS,
    plus momentum_spread_{short}_{long}d and consecutive_day_streak."""
    close = history["Close"]
    features = {}
    for n in WINDOWS:
        features[f"return_{n}d"] = simple_return(close, n)
        features[f"log_return_{n}d"] = log_return(close, n)
    for short_window, long_window in SPREAD_WINDOW_PAIRS:
        features[f"momentum_spread_{short_window}_{long_window}d"] = momentum_spread(
            close, short_window, long_window
        )
    features["consecutive_day_streak"] = consecutive_day_streak(close)
    return pd.DataFrame(features, index=history.index)
