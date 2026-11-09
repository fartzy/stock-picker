"""Calendar features: day-of-week, day-of-month, month, and day-of-week seasonality,
derived from the date index.
"""

from __future__ import annotations

import pandas as pd


def day_of_week_seasonality(history: pd.DataFrame) -> pd.Series:
    """Expanding mean close-to-close return for this date's day-of-week, using all
    prior + current occurrences of that weekday up to this row.

    Month-level seasonality is skipped for now -- with 1 year of history there are
    only ~21 same-month observations per ticker, not enough to be meaningful yet.
    """
    daily_return = history["Close"].pct_change()
    day_of_week = history.index.dayofweek
    return daily_return.groupby(day_of_week).transform(lambda s: s.expanding().mean())


def build_calendar_features(history: pd.DataFrame) -> pd.DataFrame:
    index = history.index
    return pd.DataFrame(
        {
            "day_of_week": index.dayofweek,
            "day_of_month": index.day,
            "month": index.month,
            "day_of_week_seasonality": day_of_week_seasonality(history),
        },
        index=index,
    )
