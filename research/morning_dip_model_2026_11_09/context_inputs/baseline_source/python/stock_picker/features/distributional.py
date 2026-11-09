"""Distributional features: rolling skew/kurtosis, drawdown, autocorrelation, Sharpe, VaR."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [20, 60]
TRADING_DAYS_PER_YEAR = 252
VAR_CONFIDENCE = 0.05


def rolling_skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).skew()


def rolling_kurtosis(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).kurt()


def rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Current drawdown from the trailing `window`-day peak (<= 0)."""
    rolling_peak = close.rolling(window, min_periods=1).max()
    return close / rolling_peak - 1


def return_autocorrelation(returns: pd.Series, window: int, lag: int = 1) -> pd.Series:
    return returns.rolling(window).corr(returns.shift(lag))


def rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_value_at_risk(
    returns: pd.Series, window: int, confidence: float = VAR_CONFIDENCE
) -> pd.Series:
    return returns.rolling(window).quantile(confidence)


def build_distributional_features(history: pd.DataFrame) -> pd.DataFrame:
    close = history["Close"]
    returns = close.pct_change()
    features = {}
    for n in WINDOWS:
        features[f"skew_{n}d"] = rolling_skew(returns, n)
        features[f"kurtosis_{n}d"] = rolling_kurtosis(returns, n)
        features[f"max_drawdown_{n}d"] = rolling_drawdown(close, n)
        features[f"return_autocorr_{n}d"] = return_autocorrelation(returns, n)
        features[f"sharpe_{n}d"] = rolling_sharpe(returns, n)
        features[f"var_5pct_{n}d"] = rolling_value_at_risk(returns, n)
    return pd.DataFrame(features, index=history.index)
