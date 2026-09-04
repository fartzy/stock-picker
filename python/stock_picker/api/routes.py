"""Thin JSON serving layer over the existing pipeline -- every endpoint here just
wraps an already-tested pure function from `features/`. No new business logic.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from stock_picker.features.catalog import (
    correlation_matrix,
    coverage_report,
    describe_all,
    list_feature_columns,
    top_correlated_pairs,
)
from stock_picker.features.registry import build_registry
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore

router = APIRouter(prefix="/api")


def _active_tickers() -> list[str]:
    return UniverseStore().active_tickers()


def _sample_history():
    tickers = _active_tickers()
    return PriceStore().read(tickers[0])


def _feature_tables() -> dict:
    tickers = _active_tickers()
    feature_store = FeatureStore()
    return {ticker: feature_store.read(ticker) for ticker in tickers}


@router.get("/catalog")
def get_catalog() -> dict:
    sample_history = _sample_history()
    return {
        "catalog": list_feature_columns(sample_history),
        "descriptions": describe_all(sample_history),
    }


@router.get("/coverage")
def get_coverage() -> dict:
    report = coverage_report(_feature_tables())
    return {"coverage": report["non_null_pct"].to_dict()}


@router.get("/correlation")
def get_correlation() -> dict:
    corr = correlation_matrix(_feature_tables())
    return {
        "columns": list(corr.columns),
        "matrix": corr.where(corr.notna(), None).values.tolist(),
        "top_pairs": top_correlated_pairs(corr),
    }


@router.get("/registry")
def get_registry() -> dict:
    feature_views, feature_services = build_registry(_sample_history())
    return {
        "entities": [
            {
                "name": "ticker",
                "description": (
                    "A single publicly traded stock ticker (e.g. AAPL) -- the join "
                    "key every feature view is keyed on."
                ),
            }
        ],
        "feature_views": [asdict(view) for view in feature_views],
        "feature_services": [asdict(service) for service in feature_services],
    }
