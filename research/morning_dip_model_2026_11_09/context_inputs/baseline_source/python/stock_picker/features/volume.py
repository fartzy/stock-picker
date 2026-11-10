"""Volume features: relative volume, dollar volume, OBV."""

from __future__ import annotations

import numpy as np
import pandas as pd

VOLUME_RATIO_WINDOWS = [10, 20, 60]
OBV_CHANGE_WINDOW = 20
VOLUME_ZSCORE_WINDOW = 20


def volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    return volume / volume.rolling(window).mean()


def dollar_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    return close * volume


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def obv_change(obv: pd.Series, window: int = OBV_CHANGE_WINDOW) -> pd.Series:
    return obv.diff(window)


def volume_zscore(volume: pd.Series, window: int = VOLUME_ZSCORE_WINDOW) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std


def build_volume_features(history: pd.DataFrame) -> pd.DataFrame:
    close = history["Close"]
    volume = history["Volume"]
    features = {}
    for n in VOLUME_RATIO_WINDOWS:
        features[f"volume_ratio_{n}d"] = volume_ratio(volume, n)
    features["dollar_volume"] = dollar_volume(close, volume)
    obv = on_balance_volume(close, volume)
    features["obv"] = obv
    features["obv_change_20d"] = obv_change(obv)
    features["volume_zscore_20d"] = volume_zscore(volume)
    return pd.DataFrame(features, index=history.index)
