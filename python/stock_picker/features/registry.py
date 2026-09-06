"""A lightweight feature registry, modeled on Feast's object vocabulary (Entity,
FeatureView, FeatureService) -- metadata over the existing pipeline, not a new
storage backend. `storage.feature_store.FeatureStore` remains the actual values
store; this just describes it the way a real feature store's registry would.

Derived from the real pipeline (catalog.list_feature_columns +
descriptions.describe_feature), not a hand-maintained object list that can drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from stock_picker.features.catalog import list_feature_columns

TICKER_ENTITY_NAME = "ticker"

# How many days old a feature snapshot can be before it's untrustworthy for
# inference -- every view here depends on the prior trading day's close, so a
# snapshot older than that is stale. This is the concept the pipeline was missing
# when a live inference run nearly used an in-progress row as a finalized one.
DEFAULT_TTL_DAYS = 1

_SOURCES = {
    "cross_sectional": "PriceStore + SPY benchmark + universe peer returns",
}
_DEFAULT_SOURCE = "PriceStore"

_TAGS = {
    "cross_sectional": {"cross_ticker": "true"},
}

OWNER = "stock-picker"


@dataclass
class Entity:
    name: str
    description: str


@dataclass
class FeatureView:
    name: str
    entities: list[str]
    features: list[str]
    source: str
    ttl_days: int
    tags: dict[str, str] = field(default_factory=dict)
    owner: str = OWNER


@dataclass
class FeatureService:
    name: str
    feature_views: list[str]
    description: str


@dataclass
class FreshnessResult:
    ok: bool
    age_days: int
    ttl_days: int


TICKER_ENTITY = Entity(
    name=TICKER_ENTITY_NAME,
    description="A single publicly traded stock ticker (e.g. AAPL), the join key every feature view is keyed on.",
)


def build_registry(sample_history: pd.DataFrame) -> tuple[list[FeatureView], list[FeatureService]]:
    """Assemble the registry from the real pipeline: one FeatureView per category
    in catalog.list_feature_columns, plus the one FeatureService that consumes all
    of them (today's single model, day_session_return)."""
    catalog = list_feature_columns(sample_history)

    feature_views = [
        FeatureView(
            name=category,
            entities=[TICKER_ENTITY_NAME],
            features=columns,
            source=_SOURCES.get(category, _DEFAULT_SOURCE),
            ttl_days=DEFAULT_TTL_DAYS,
            tags=_TAGS.get(category, {}),
        )
        for category, columns in catalog.items()
    ]

    feature_services = [
        FeatureService(
            name="day_session_return_model",
            feature_views=[view.name for view in feature_views],
            description="Feeds training.dataset.build_training_frame for the day-session (open->close) return model.",
        )
    ]

    return feature_views, feature_services


def check_freshness(ttl_days: int, snapshot_date: date, as_of_date: date) -> FreshnessResult:
    """Is `snapshot_date` (the date a feature snapshot is from) fresh enough to use
    for inference as of `as_of_date`, given `ttl_days`?

    Takes a bare `ttl_days` rather than a whole `FeatureView` -- freshness only ever
    depends on that one field, and every view uses the same `DEFAULT_TTL_DAYS` today
    anyway (inference.py's row spans every category at once, so there's no single
    "the" view to hand this function regardless).
    """
    age_days = (as_of_date - snapshot_date).days
    return FreshnessResult(ok=age_days <= ttl_days, age_days=age_days, ttl_days=ttl_days)
