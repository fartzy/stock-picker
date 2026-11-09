"""Oscillator features: RSI, Stochastic, Bollinger Bands, Williams %R, CCI."""

from __future__ import annotations

import pandas as pd

RSI_WINDOWS = [2, 14, 28]  # RSI-2 is a well-known short-term mean-reversion variant
STOCHASTIC_WINDOWS = [5, 14]
STOCHASTIC_SIGNAL_WINDOW = 3
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2
WILLIAMS_R_WINDOW = 14
CCI_WINDOW = 20
# Lambert's original 1980 CCI scaling constant -- fixed across every standard
# CCI implementation (same role as BOLLINGER_NUM_STD above: a named part of
# the formula's own definition, not something to tune per-call).
CCI_SCALING_CONSTANT = 0.015


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def stochastic_oscillator(
    history: pd.DataFrame,
    window: int,
    signal_window: int = STOCHASTIC_SIGNAL_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = history["Low"].rolling(window).min()
    highest_high = history["High"].rolling(window).max()
    percent_k = 100 * (history["Close"] - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(signal_window).mean()
    return percent_k, percent_d


def bollinger_bands(
    close: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_NUM_STD,
) -> tuple[pd.Series, pd.Series]:
    """Returns (%B, bandwidth)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    percent_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return percent_b, bandwidth


def williams_r(history: pd.DataFrame, window: int = WILLIAMS_R_WINDOW) -> pd.Series:
    highest_high = history["High"].rolling(window).max()
    lowest_low = history["Low"].rolling(window).min()
    return -100 * (highest_high - history["Close"]) / (highest_high - lowest_low)


def commodity_channel_index(history: pd.DataFrame, window: int = CCI_WINDOW) -> pd.Series:
    typical_price = (history["High"] + history["Low"] + history["Close"]) / 3
    sma_tp = typical_price.rolling(window).mean()
    mean_deviation = typical_price.rolling(window).apply(
        lambda x: abs(x - x.mean()).mean(), raw=True
    )
    return (typical_price - sma_tp) / (CCI_SCALING_CONSTANT * mean_deviation)


def build_oscillator_features(history: pd.DataFrame) -> pd.DataFrame:
    close = history["Close"]
    features = {}
    for n in RSI_WINDOWS:
        features[f"rsi_{n}"] = rsi(close, n)
    for n in STOCHASTIC_WINDOWS:
        percent_k, percent_d = stochastic_oscillator(history, window=n)
        features[f"stochastic_k_{n}d"] = percent_k
        features[f"stochastic_d_{n}d"] = percent_d
    percent_b, bandwidth = bollinger_bands(close)
    features["bollinger_pct_b"] = percent_b
    features["bollinger_bandwidth"] = bandwidth
    features["williams_r"] = williams_r(history)
    features["cci"] = commodity_channel_index(history)
    return pd.DataFrame(features, index=history.index)
