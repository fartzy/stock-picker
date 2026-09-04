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
from stock_picker.features.catalog_loader import feature_tables, sample_history
from stock_picker.features.registry import build_registry

router = APIRouter(prefix="/api")


@router.get("/catalog")
def get_catalog() -> dict:
    history = sample_history()
    return {
        "catalog": list_feature_columns(history),
        "descriptions": describe_all(history),
    }


@router.get("/coverage")
def get_coverage() -> dict:
    report = coverage_report(feature_tables())
    return {"coverage": report["non_null_pct"].to_dict()}


@router.get("/correlation")
def get_correlation() -> dict:
    corr = correlation_matrix(feature_tables())
    return {
        "columns": list(corr.columns),
        "matrix": corr.where(corr.notna(), None).values.tolist(),
        "top_pairs": top_correlated_pairs(corr),
    }


@router.get("/registry")
def get_registry() -> dict:
    feature_views, feature_services = build_registry(sample_history())
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
