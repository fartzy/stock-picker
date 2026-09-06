"""Pydantic request/response models for the FastAPI JSON layer.

Kept separate from routes.py so route handlers stay thin wiring -- every model
here mirrors, field-for-field, both a `features/`-layer return shape and its
`typescript/src/api.ts` counterpart. Response models were added after routes.py
had shipped for a while returning bare `dict`s -- this finishes that, giving
FastAPI's auto-generated OpenAPI schema and response validation for free,
matching the pydantic convention already used for request bodies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from stock_picker.training.ensemble import EnsembleMemberInfo

# ---- requests ----


class TradeCreate(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    shares: float
    price: float


class PruneRequest(BaseModel):
    reason: str | None = None


class FeatureSelectionRequest(BaseModel):
    included_features: list[str]


class ModelChoice(BaseModel):
    model_type: str
    weight: float = 1.0


class ModelSelectionRequest(BaseModel):
    model_choices: list[ModelChoice]


# ---- responses ----


class CatalogResponse(BaseModel):
    catalog: dict[str, list[str]]
    descriptions: dict[str, str]
    formulas: dict[str, str]


class CoverageResponse(BaseModel):
    coverage: dict[str, float]


class CorrelationPair(BaseModel):
    a: str
    b: str
    correlation: float


class CorrelationResponse(BaseModel):
    columns: list[str]
    matrix: list[list[float | None]]
    top_pairs: list[CorrelationPair]


class Trade(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    shares: float
    price: float
    notional: float
    executed_at: str
    # None for a "buy" row -- only a closing "sell" has a realized P&L.
    realized_pnl: float | None = None


class TradesResponse(BaseModel):
    trades: list[Trade]


class Position(BaseModel):
    ticker: str
    day: str
    shares: float
    invested: float
    buy_time: str | None
    buy_price: float | None
    day_open: float | None
    prev_close: float | None
    gap: float | None
    gap_pct: float | None
    sell_time: str | None
    sell_price: float | None
    current_price: float | None
    closed: bool
    pnl: float | None


class PositionsResponse(BaseModel):
    positions: list[Position]


class QuoteSummary(BaseModel):
    ticker: str
    open: float
    last: float
    diff: float
    diff_pct: float
    prev_close: float | None = None
    gap: float | None = None
    gap_pct: float | None = None


class QuotesResponse(BaseModel):
    quotes: list[QuoteSummary]


class PrunedFeatureEntry(BaseModel):
    feature: str
    reason: str
    pruned_at: str


class PrunedFeaturesResponse(BaseModel):
    pruned_features: list[str]
    archive: list[PrunedFeatureEntry]


class FeatureSelectionResponse(BaseModel):
    # None = no explicit selection -- every feature, subject to pruning only.
    included_features: list[str] | None


class ModelSelectionResponse(BaseModel):
    # None = no explicit choice -- training/main.py's own DEFAULT_MODEL_SPECS.
    model_choices: list[ModelChoice] | None
    # Which model types the composable picker offers. logistic_regression is
    # deliberately excluded -- it's fit as a standalone diagnostic (see
    # training/main.py), never an ensemble member (see model.py's
    # PREDICTIVE_MODEL_TYPES).
    available_model_types: list[str]


class ImportanceResponse(BaseModel):
    importance: dict[str, float]
    by_model_type: dict[str, dict[str, float]]


class ModelInfoResponse(BaseModel):
    models: list[EnsembleMemberInfo]


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceHistoryResponse(BaseModel):
    ticker: str
    interval: Literal["daily", "hourly"]
    prices: list[PricePoint]


class Entity(BaseModel):
    name: str
    description: str


class FeatureView(BaseModel):
    name: str
    entities: list[str]
    features: list[str]
    source: str
    ttl_days: int
    tags: dict[str, str]
    owner: str


class FeatureService(BaseModel):
    name: str
    feature_views: list[str]
    description: str


class RegistryResponse(BaseModel):
    entities: list[Entity]
    feature_views: list[FeatureView]
    feature_services: list[FeatureService]
