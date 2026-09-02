"""Orchestrates all feature categories into a single per-ticker feature DataFrame."""

from __future__ import annotations

import pandas as pd

from stock_picker.features.calendar import build_calendar_features
from stock_picker.features.candle import build_candle_features
from stock_picker.features.cross_sectional import (
    RETURN_RANK_WINDOWS,
    build_cross_sectional_features,
    return_rank,
)
from stock_picker.features.distributional import build_distributional_features
from stock_picker.features.momentum import build_momentum_features
from stock_picker.features.oscillators import build_oscillator_features
from stock_picker.features.trend import build_trend_features
from stock_picker.features.volatility import build_volatility_features
from stock_picker.features.volume import build_volume_features


def build_features(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame | None = None,
    peer_return_ranks: dict[int, pd.Series] | None = None,
    sector_avg_return: pd.Series | None = None,
) -> pd.DataFrame:
    """Combine every feature category for a single ticker's OHLCV history.

    Cross-sectional inputs are optional -- see `build_cross_sectional_features` for
    what gets omitted when they aren't supplied.
    """
    categories = [
        build_momentum_features(history),
        build_volatility_features(history),
        build_trend_features(history),
        build_oscillator_features(history),
        build_volume_features(history),
        build_candle_features(history),
        build_distributional_features(history),
        build_calendar_features(history),
        build_cross_sectional_features(
            history,
            benchmark_history=benchmark_history,
            peer_return_ranks=peer_return_ranks,
            sector_avg_return=sector_avg_return,
        ),
    ]
    return pd.concat(categories, axis=1)


def build_features_for_universe(
    histories: dict[str, pd.DataFrame],
    benchmark_history: pd.DataFrame | None = None,
    sector_by_ticker: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute features for every ticker in `histories`, including the cross-sectional
    return-rank columns that require the whole universe's returns together.
    """
    n_day_returns = {
        window: pd.DataFrame(
            {
                ticker: history["Close"].pct_change(window)
                for ticker, history in histories.items()
            }
        )
        for window in RETURN_RANK_WINDOWS
    }
    rank_by_window = {window: return_rank(returns) for window, returns in n_day_returns.items()}

    sector_avg_returns = None
    if sector_by_ticker:
        daily_returns = pd.DataFrame(
            {ticker: history["Close"].pct_change() for ticker, history in histories.items()}
        )
        sector_avg_returns = {
            sector: daily_returns[
                [t for t, s in sector_by_ticker.items() if s == sector and t in daily_returns]
            ].mean(axis=1)
            for sector in set(sector_by_ticker.values())
        }

    features_by_ticker = {}
    for ticker, history in histories.items():
        peer_return_ranks = {window: ranks[ticker] for window, ranks in rank_by_window.items()}
        sector_avg_return = None
        if sector_avg_returns and ticker in (sector_by_ticker or {}):
            sector_avg_return = sector_avg_returns[sector_by_ticker[ticker]]

        features_by_ticker[ticker] = build_features(
            history,
            benchmark_history=benchmark_history,
            peer_return_ranks=peer_return_ranks,
            sector_avg_return=sector_avg_return,
        )

    return features_by_ticker
