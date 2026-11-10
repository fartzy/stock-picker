"""Point-in-time market breadth from the tracked universe's completed bars.

A name contributes only on or after its registry first_seen date. This avoids
projecting today's tracked universe backward into a period when it was not
tracked; missing pre-registry history remains missing rather than zero.
"""

from __future__ import annotations

import pandas as pd

MIN_BREADTH_NAMES = 40
BREADTH_COLUMNS = (
    "market_advance_fraction",
    "market_above_sma_20d_fraction",
    "market_return_dispersion_1d",
    "market_up_volume_fraction",
)


def _eligible(series: pd.Series, first_seen: str) -> pd.Series:
    return series.where(series.index >= pd.Timestamp(first_seen))


def build_breadth_features(
    histories: dict[str, pd.DataFrame], first_seen_by_ticker: dict[str, str]
) -> pd.DataFrame:
    """Compute completed-session breadth, requiring enough observed members.

    The result is one shared daily series, attached to each ticker's feature
    snapshot and shifted one session in the training frame like other close
    features. No price or membership from a later date enters a given row.
    """
    returns, above_sma, volumes = {}, {}, {}
    for ticker, history in histories.items():
        first_seen = first_seen_by_ticker.get(ticker)
        if first_seen is None:
            continue
        close = history["Close"]
        daily_return = close.pct_change(fill_method=None)
        sma_20 = close.rolling(20).mean()
        returns[ticker] = _eligible(daily_return, first_seen)
        above_sma[ticker] = _eligible(close.gt(sma_20).where(sma_20.notna()), first_seen)
        volumes[ticker] = _eligible(history["Volume"].where(history["Volume"].gt(0)), first_seen)

    if not returns:
        return pd.DataFrame(columns=BREADTH_COLUMNS)

    daily = pd.DataFrame(returns)
    above = pd.DataFrame(above_sma)
    volume = pd.DataFrame(volumes).where(daily.notna())
    eligible_count = daily.count(axis=1)
    above_count = above.count(axis=1)
    volume_total = volume.sum(axis=1, min_count=1)
    up_volume = volume.where(daily.gt(0)).sum(axis=1, min_count=1).fillna(0)

    return pd.DataFrame(
        {
            "market_advance_fraction": daily.gt(0).where(daily.notna()).mean(axis=1).where(
                eligible_count.ge(MIN_BREADTH_NAMES)
            ),
            "market_above_sma_20d_fraction": above.mean(axis=1).where(
                above_count.ge(MIN_BREADTH_NAMES)
            ),
            "market_return_dispersion_1d": daily.std(axis=1).where(
                eligible_count.ge(MIN_BREADTH_NAMES)
            ),
            "market_up_volume_fraction": (up_volume / volume_total).where(
                eligible_count.ge(MIN_BREADTH_NAMES) & volume_total.gt(0)
            ),
        }
    )
