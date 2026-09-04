"""Introspection: what features exist, grouped by category, and how well-populated
they are in real persisted data. Pure functions only -- no storage dependency, so
this stays trivially testable. See catalog_main.py for the runnable entrypoint that
wires this to PriceStore/FeatureStore.
"""

from __future__ import annotations

import pandas as pd

from stock_picker.features import (
    calendar,
    candle,
    cross_sectional,
    distributional,
    momentum,
    oscillators,
    trend,
    volatility,
    volume,
)
from stock_picker.features.descriptions import describe_feature

_SINGLE_TICKER_BUILDERS = {
    "momentum": momentum.build_momentum_features,
    "volatility": volatility.build_volatility_features,
    "trend": trend.build_trend_features,
    "oscillators": oscillators.build_oscillator_features,
    "volume": volume.build_volume_features,
    "candle": candle.build_candle_features,
    "distributional": distributional.build_distributional_features,
    "calendar": calendar.build_calendar_features,
}


def _cross_sectional_columns(sample_history: pd.DataFrame) -> list[str]:
    """Force every cross-sectional column to materialize (dummy benchmark/peer/
    sector inputs) so the catalog reflects the full design, including
    sector_relative_return -- which is real code but not populated in production
    yet, since we don't persist sector labels."""
    dummy_rank = pd.Series(0.5, index=sample_history.index)
    features = cross_sectional.build_cross_sectional_features(
        sample_history,
        benchmark_history=sample_history,
        peer_return_ranks={w: dummy_rank for w in cross_sectional.RETURN_RANK_WINDOWS},
        sector_avg_return=sample_history["Close"].pct_change(),
    )
    return list(features.columns)


def list_feature_columns(sample_history: pd.DataFrame) -> dict[str, list[str]]:
    """Category -> column names, derived by actually calling each builder so this
    can't drift from what pipeline.py really produces."""
    columns = {
        category: list(builder(sample_history).columns)
        for category, builder in _SINGLE_TICKER_BUILDERS.items()
    }
    columns["cross_sectional"] = _cross_sectional_columns(sample_history)
    return columns


def describe_all(sample_history: pd.DataFrame) -> dict[str, str]:
    """Feature name -> plain-English description, for every real column."""
    catalog = list_feature_columns(sample_history)
    return {
        column: describe_feature(column)
        for columns in catalog.values()
        for column in columns
    }


def coverage_report(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Non-null percentage per feature column, averaged across `feature_tables`
    (ticker -> its persisted feature DataFrame), sorted ascending -- surfaces a
    formula bug or a too-long window before it silently produces an all-NaN column.
    """
    non_null_pct = pd.concat(
        [table.notna().mean() for table in feature_tables.values()], axis=1
    ).mean(axis=1)
    return non_null_pct.sort_values().rename("non_null_pct").to_frame()
