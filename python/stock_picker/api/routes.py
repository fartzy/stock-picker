"""Thin JSON serving layer over the existing pipeline -- every endpoint here just
wraps an already-tested pure function from `features/`. No new business logic.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException

from stock_picker.api.models import (
    CatalogResponse,
    CorrelationResponse,
    CoverageResponse,
    FeatureSelectionRequest,
    FeatureSelectionResponse,
    ImportanceResponse,
    ModelInfoResponse,
    ModelSelectionRequest,
    ModelSelectionResponse,
    PositionsResponse,
    PriceHistoryResponse,
    PruneRequest,
    PrunedFeaturesResponse,
    QuotesResponse,
    RegistryResponse,
    TradeCreate,
    TradesResponse,
)
from stock_picker.api.models import ModelChoice as ModelChoiceModel
from stock_picker.features.catalog import (
    compute_formulas_all,
    correlation_matrix,
    coverage_report,
    describe_all,
    list_feature_columns,
    top_correlated_pairs,
)
from stock_picker.features.catalog_loader import feature_tables, sample_history
from stock_picker.features.price_history import (
    daily_price_history,
    intraday_price_history,
    price_series,
)
from stock_picker.features.pruning import pruned_features
from stock_picker.features.quotes import fetch_ticker_quotes, quote_summaries
from stock_picker.features.registry import TICKER_ENTITY, build_registry
from stock_picker.features.selection import selected_features
from stock_picker.features.trades import position_summaries, trade_history, trade_log
from stock_picker.storage.feature_exclusion_store import DEFAULT_REASON, PrunedFeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.trade_store import Trade, TradeStore
from stock_picker.storage.training_config_store import ModelChoice, TrainingConfigStore
from stock_picker.training import job as training_job
from stock_picker.training.ensemble import ensemble_composition, selected_model_specs
from stock_picker.training.importance import model_importance
from stock_picker.training.job import JobStatus
from stock_picker.training.main import MODEL_NAME
from stock_picker.training.model import PREDICTIVE_MODEL_TYPES

router = APIRouter(prefix="/api")


@router.get("/catalog")
def get_catalog() -> CatalogResponse:
    history = sample_history()
    return CatalogResponse(
        catalog=list_feature_columns(history),
        descriptions=describe_all(history),
        formulas=compute_formulas_all(history),
    )


@router.get("/coverage")
def get_coverage() -> CoverageResponse:
    report = coverage_report(feature_tables())
    return CoverageResponse(coverage=report["non_null_pct"].to_dict())


@router.get("/correlation")
def get_correlation() -> CorrelationResponse:
    corr = correlation_matrix(feature_tables())
    return CorrelationResponse(
        columns=list(corr.columns),
        matrix=corr.where(corr.notna(), None).values.tolist(),
        top_pairs=top_correlated_pairs(corr),
    )


@router.get("/trades")
def get_trades() -> TradesResponse:
    return TradesResponse(trades=trade_history(trade_log()))


@router.post("/trades")
def create_trade(trade: TradeCreate) -> TradesResponse:
    TradeStore().append(
        Trade(
            ticker=trade.ticker,
            side=trade.side,
            shares=trade.shares,
            price=trade.price,
            executed_at=datetime.now().astimezone().isoformat(),
        )
    )
    return TradesResponse(trades=trade_history(trade_log()))


@router.get("/quotes")
def get_quotes(tickers: str) -> QuotesResponse:
    return QuotesResponse(quotes=quote_summaries(fetch_ticker_quotes(tickers.split(","))))


@router.get("/positions")
def get_positions() -> PositionsResponse:
    trades = trade_log()
    tickers = sorted(trades["ticker"].unique()) if not trades.empty else []
    quotes = fetch_ticker_quotes(tickers) if tickers else {}
    return PositionsResponse(positions=position_summaries(trades, quotes))


@router.get("/pruned-features")
def get_pruned_features() -> PrunedFeaturesResponse:
    return PrunedFeaturesResponse(
        pruned_features=sorted(pruned_features()),
        archive=PrunedFeatureStore().read_all(),
    )


@router.post("/features/{feature}/prune")
def prune_feature(feature: str, body: PruneRequest | None = None) -> PrunedFeaturesResponse:
    reason = (body.reason if body else None) or DEFAULT_REASON
    PrunedFeatureStore().prune(feature, reason=reason)
    return PrunedFeaturesResponse(
        pruned_features=sorted(pruned_features()),
        archive=PrunedFeatureStore().read_all(),
    )


@router.delete("/features/{feature}/prune")
def unprune_feature(feature: str) -> PrunedFeaturesResponse:
    PrunedFeatureStore().unprune(feature)
    return PrunedFeaturesResponse(
        pruned_features=sorted(pruned_features()),
        archive=PrunedFeatureStore().read_all(),
    )


@router.get("/feature-importance")
def get_feature_importance() -> ImportanceResponse:
    importance = model_importance()
    return ImportanceResponse(importance=importance["blended"], by_model_type=importance["by_model_type"])


@router.get("/model-info")
def get_model_info() -> ModelInfoResponse:
    store = ModelStore()
    if not store.exists(MODEL_NAME):
        return ModelInfoResponse(models=[])
    return ModelInfoResponse(models=ensemble_composition(store.read(MODEL_NAME)))


@router.get("/feature-selection")
def get_feature_selection() -> FeatureSelectionResponse:
    features = selected_features()
    return FeatureSelectionResponse(included_features=sorted(features) if features is not None else None)


@router.post("/feature-selection")
def set_feature_selection(body: FeatureSelectionRequest) -> FeatureSelectionResponse:
    TrainingConfigStore().write_included_features(set(body.included_features))
    return FeatureSelectionResponse(included_features=sorted(body.included_features))


@router.delete("/feature-selection")
def clear_feature_selection() -> FeatureSelectionResponse:
    TrainingConfigStore().write_included_features(None)
    return FeatureSelectionResponse(included_features=None)


@router.get("/model-selection")
def get_model_selection() -> ModelSelectionResponse:
    choices = TrainingConfigStore().read().model_choices
    return ModelSelectionResponse(
        model_choices=(
            [ModelChoiceModel(model_type=c.model_type, weight=c.weight) for c in choices]
            if choices is not None
            else None
        ),
        available_model_types=PREDICTIVE_MODEL_TYPES,
    )


@router.post("/model-selection")
def set_model_selection(body: ModelSelectionRequest) -> ModelSelectionResponse:
    choices = [ModelChoice(model_type=c.model_type, weight=c.weight) for c in body.model_choices]
    TrainingConfigStore().write_model_choices(choices)
    return ModelSelectionResponse(
        model_choices=[ModelChoiceModel(model_type=c.model_type, weight=c.weight) for c in choices],
        available_model_types=PREDICTIVE_MODEL_TYPES,
    )


@router.delete("/model-selection")
def clear_model_selection() -> ModelSelectionResponse:
    TrainingConfigStore().write_model_choices(None)
    return ModelSelectionResponse(model_choices=None, available_model_types=PREDICTIVE_MODEL_TYPES)


@router.post("/training/run")
def start_training_run() -> JobStatus:
    started = training_job.start(included_features=selected_features(), model_specs=selected_model_specs())
    if not started:
        raise HTTPException(status_code=409, detail="a training run is already in progress")
    return training_job.status()


@router.get("/training/status")
def get_training_status() -> JobStatus:
    return training_job.status()


@router.get("/prices/{ticker}")
def get_price_history(ticker: str, interval: Literal["daily", "hourly"] = "daily") -> PriceHistoryResponse:
    try:
        history = intraday_price_history(ticker) if interval == "hourly" else daily_price_history(ticker)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"no {interval} price history for {ticker}")
    return PriceHistoryResponse(ticker=ticker, interval=interval, prices=price_series(history))


@router.get("/registry")
def get_registry() -> RegistryResponse:
    feature_views, feature_services = build_registry(sample_history())
    return RegistryResponse(
        entities=[asdict(TICKER_ENTITY)],
        feature_views=[asdict(view) for view in feature_views],
        feature_services=[asdict(service) for service in feature_services],
    )
