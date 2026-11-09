"""Cross-sectional features: rank vs peers, beta/correlation/relative-strength vs a
benchmark, sector-relative return.

Unlike the other category modules, these need data beyond a single ticker's own OHLCV
history -- the rest of the universe's returns (for ranking) and a benchmark series (e.g.
SPY) for beta/correlation/relative-strength. `build_cross_sectional_features` accepts
those as plain inputs and never fetches anything itself; any input that isn't supplied
just means its columns are omitted rather than filled with NaN.
"""

from __future__ import annotations

import pandas as pd

RETURN_RANK_WINDOWS = [5, 20]
BETA_WINDOW = 60
CORRELATION_WINDOW = 60


def return_rank(returns_by_ticker: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile rank (0-1) of each ticker's return among all columns."""
    return returns_by_ticker.rank(axis=1, pct=True)


def rolling_beta(
    ticker_returns: pd.Series, benchmark_returns: pd.Series, window: int = BETA_WINDOW
) -> pd.Series:
    covariance = ticker_returns.rolling(window).cov(benchmark_returns)
    benchmark_variance = benchmark_returns.rolling(window).var()
    return covariance / benchmark_variance


def rolling_correlation(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = CORRELATION_WINDOW,
) -> pd.Series:
    return ticker_returns.rolling(window).corr(benchmark_returns)


def relative_strength(ticker_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """Cumulative ticker return minus cumulative benchmark return, both rebased to the
    start of the given series."""
    ticker_cum = ticker_close / ticker_close.iloc[0] - 1
    benchmark_cum = benchmark_close / benchmark_close.iloc[0] - 1
    return ticker_cum - benchmark_cum


def sector_relative_return(ticker_return: pd.Series, sector_avg_return: pd.Series) -> pd.Series:
    return ticker_return - sector_avg_return


def build_cross_sectional_features(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame | None = None,
    peer_return_ranks: dict[int, pd.Series] | None = None,
    sector_avg_return: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble cross-sectional columns from precomputed inputs.

    `peer_return_ranks` maps window -> this ticker's percentile-rank series, already
    extracted from a universe-wide `return_rank` call (ranking needs every ticker's
    returns together, not just this one's history -- see
    `pipeline.build_features_for_universe`).
    """
    close = history["Close"]
    daily_return = close.pct_change()
    features: dict[str, pd.Series] = {}

    if peer_return_ranks:
        for window, rank_series in peer_return_ranks.items():
            features[f"return_rank_{window}d"] = rank_series

    if benchmark_history is not None:
        benchmark_close = benchmark_history["Close"]
        benchmark_return = benchmark_close.pct_change()
        features["beta_60d"] = rolling_beta(daily_return, benchmark_return)
        features["correlation_60d"] = rolling_correlation(daily_return, benchmark_return)
        features["relative_strength"] = relative_strength(close, benchmark_close)

    if sector_avg_return is not None:
        features["sector_relative_return"] = sector_relative_return(
            daily_return, sector_avg_return
        )

    return pd.DataFrame(features, index=history.index)
