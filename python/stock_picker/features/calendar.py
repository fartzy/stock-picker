"""Calendar features: day-of-week, day-of-month, month, derived from the date index."""

from __future__ import annotations

import pandas as pd


def build_calendar_features(history: pd.DataFrame) -> pd.DataFrame:
    index = history.index
    return pd.DataFrame(
        {
            "day_of_week": index.dayofweek,
            "day_of_month": index.day,
            "month": index.month,
        },
        index=index,
    )
