"""Momentum/return features: simple and log returns over multiple windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [1, 3, 5, 10, 20, 60, 120]


def simple_return(close: pd.Series, window: int) -> pd.Series:
    """Percent change over `window` trailing trading days."""
    return close.pct_change(window)


def log_return(close: pd.Series, window: int) -> pd.Series:
    """Log return over `window` trailing trading days."""
    return np.log(close / close.shift(window))


def build_momentum_features(history: pd.DataFrame) -> pd.DataFrame:
    """Return simple_return_{n}d and log_return_{n}d for each window in WINDOWS."""
    close = history["Close"]
    features = {}
    for n in WINDOWS:
        features[f"return_{n}d"] = simple_return(close, n)
        features[f"log_return_{n}d"] = log_return(close, n)
    return pd.DataFrame(features, index=history.index)
