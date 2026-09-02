"""Deterministic synthetic OHLCV fixture shared across feature tests."""

from __future__ import annotations

import math

import pandas as pd


def synthetic_history(n: int = 40) -> pd.DataFrame:
    """A reproducible, no-randomness OHLCV series with both up and down moves."""
    dates = pd.bdate_range("2026-01-02", periods=n)
    closes = [100.0 + 0.3 * i + 3.0 * math.sin(i / 4) for i in range(n)]
    opens = [closes[i - 1] + 0.2 * math.cos(i) if i > 0 else closes[0] - 0.1 for i in range(n)]
    highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    volumes = [1_000_000 + 10_000 * (i % 7) for i in range(n)]

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Adj Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
