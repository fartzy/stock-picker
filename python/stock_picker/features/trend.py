"""Trend features: price relative to moving averages, MACD."""

from __future__ import annotations

import pandas as pd

SMA_WINDOWS = [5, 10, 20, 50, 100]
EMA_WINDOWS = [12, 26, 50, 100]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def price_vs_sma(close: pd.Series, window: int) -> pd.Series:
    return close / sma(close, window) - 1


def price_vs_ema(close: pd.Series, span: int) -> pd.Series:
    return close / ema(close, span) - 1


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def build_trend_features(history: pd.DataFrame) -> pd.DataFrame:
    close = history["Close"]
    features = {}
    for n in SMA_WINDOWS:
        features[f"price_vs_sma_{n}d"] = price_vs_sma(close, n)
    for n in EMA_WINDOWS:
        features[f"price_vs_ema_{n}d"] = price_vs_ema(close, n)
    macd_line, macd_signal, macd_hist = macd(close)
    features["macd_line"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_hist
    return pd.DataFrame(features, index=history.index)
