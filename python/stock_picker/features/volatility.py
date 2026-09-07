"""Volatility features: rolling realized vol, ATR, Parkinson, Garman-Klass estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

REALIZED_VOL_WINDOWS = [5, 10, 20, 60, 120]
RANGE_VOL_WINDOWS = [10, 20, 60]
ATR_WINDOW = 14
# volatility_Nd's own level answers "how volatile has this stock been" --
# these lags answer a distinct question, "is that volatility currently
# expanding or contracting," which the level alone can't distinguish (a
# stock calm the whole window looks identical to one that was choppy then
# just went quiet). 20d matches the window already used elsewhere in this
# codebase (the volatility-normalized confidence threshold) as the
# reference realized-vol measure.
VOLATILITY_DELTA_BASE_WINDOW = 20
VOLATILITY_DELTA_LAGS = [1, 3, 5, 10]


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling standard deviation of daily returns."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def true_range(history: pd.DataFrame) -> pd.Series:
    prior_close = history["Close"].shift(1)
    ranges = pd.concat(
        [
            history["High"] - history["Low"],
            (history["High"] - prior_close).abs(),
            (history["Low"] - prior_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(history: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    return true_range(history).rolling(window).mean()


def parkinson_volatility(history: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson (1980) high-low range volatility estimator, annualized."""
    log_hl_sq = np.log(history["High"] / history["Low"]) ** 2
    variance = log_hl_sq.rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(variance) * np.sqrt(TRADING_DAYS_PER_YEAR)


def garman_klass_volatility(history: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator, annualized."""
    log_hl_sq = np.log(history["High"] / history["Low"]) ** 2
    log_co_sq = np.log(history["Close"] / history["Open"]) ** 2
    daily_variance = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
    variance = daily_variance.rolling(window).mean()
    return np.sqrt(variance.clip(lower=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def volatility_delta(volatility: pd.Series, lag: int) -> pd.Series:
    """Change in rolling volatility over `lag` trading days -- is realized
    volatility currently expanding or contracting, a distinct question from
    volatility_Nd's own level (how volatile has this stock been), which
    can't tell apart a stock calm the whole window from one that was choppy
    then just went quiet."""
    return volatility.diff(lag)


def build_volatility_features(history: pd.DataFrame) -> pd.DataFrame:
    daily_returns = history["Close"].pct_change()
    features = {}
    for n in REALIZED_VOL_WINDOWS:
        features[f"volatility_{n}d"] = rolling_volatility(daily_returns, n)
    features["atr_14"] = average_true_range(history)
    for n in RANGE_VOL_WINDOWS:
        features[f"parkinson_vol_{n}d"] = parkinson_volatility(history, n)
    for n in RANGE_VOL_WINDOWS:
        features[f"garman_klass_vol_{n}d"] = garman_klass_volatility(history, n)
    base_volatility = features[f"volatility_{VOLATILITY_DELTA_BASE_WINDOW}d"]
    for lag in VOLATILITY_DELTA_LAGS:
        features[f"volatility_delta_{lag}d"] = volatility_delta(base_volatility, lag)
    return pd.DataFrame(features, index=history.index)
