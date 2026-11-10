"""Market-wide overnight regime: SPY gap and VIX, known before the cash open.

One series is broadcast onto every ticker. Not a 2000-name download -- SPY
and ^VIX only. Gap uses today's SPY open vs yesterday's close (open-known).
VIX level is yesterday's close (last completed session).
"""

from __future__ import annotations

import pandas as pd

SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"


def spy_overnight_gap(spy: pd.DataFrame) -> pd.Series:
    prev_close = spy["Close"].shift(1)
    return (spy["Open"] - prev_close) / prev_close


def vix_level(vix: pd.DataFrame) -> pd.Series:
    return vix["Close"]


def build_regime_features(
    index: pd.Index,
    spy_history: pd.DataFrame | None = None,
    vix_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    if spy_history is not None and not spy_history.empty:
        gap = spy_overnight_gap(spy_history).reindex(index)
        features["spy_overnight_gap"] = gap
    if vix_history is not None and not vix_history.empty:
        features["vix_close"] = vix_level(vix_history).reindex(index)
    if not features:
        return pd.DataFrame(index=index)
    return pd.DataFrame(features, index=index)
