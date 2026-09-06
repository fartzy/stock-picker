"""Thin JSON serving layer over the existing pipeline -- every endpoint here just
wraps an already-tested pure function from `features/`. No new business logic.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
from stock_picker.features.registry import build_registry
from stock_picker.features.trades import position_summaries, trade_history, trade_log
from stock_picker.storage.feature_exclusion_store import DEFAULT_REASON, PrunedFeatureStore
from stock_picker.storage.trade_store import Trade, TradeStore
from stock_picker.training.importance import model_importance

router = APIRouter(prefix="/api")


class TradeCreate(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    shares: float
    price: float


class PruneRequest(BaseModel):
    reason: str | None = None


@router.get("/catalog")
def get_catalog() -> dict:
    history = sample_history()
    return {
        "catalog": list_feature_columns(history),
        "descriptions": describe_all(history),
        "formulas": compute_formulas_all(history),
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


@router.get("/trades")
def get_trades() -> dict:
    return {"trades": trade_history(trade_log())}


@router.post("/trades")
def create_trade(trade: TradeCreate) -> dict:
    TradeStore().append(
        Trade(
            ticker=trade.ticker,
            side=trade.side,
            shares=trade.shares,
            price=trade.price,
            executed_at=datetime.now().astimezone().isoformat(),
        )
    )
    return {"trades": trade_history(trade_log())}


@router.get("/quotes")
def get_quotes(tickers: str) -> dict:
    return {"quotes": quote_summaries(fetch_ticker_quotes(tickers.split(",")))}


@router.get("/positions")
def get_positions() -> dict:
    trades = trade_log()
    tickers = sorted(trades["ticker"].unique()) if not trades.empty else []
    quotes = fetch_ticker_quotes(tickers) if tickers else {}
    return {"positions": position_summaries(trades, quotes)}


@router.get("/pruned-features")
def get_pruned_features() -> dict:
    return {
        "pruned_features": sorted(pruned_features()),
        "archive": PrunedFeatureStore().read_all(),
    }


@router.post("/features/{feature}/prune")
def prune_feature(feature: str, body: PruneRequest | None = None) -> dict:
    reason = (body.reason if body else None) or DEFAULT_REASON
    PrunedFeatureStore().prune(feature, reason=reason)
    return {
        "pruned_features": sorted(pruned_features()),
        "archive": PrunedFeatureStore().read_all(),
    }


@router.delete("/features/{feature}/prune")
def unprune_feature(feature: str) -> dict:
    PrunedFeatureStore().unprune(feature)
    return {
        "pruned_features": sorted(pruned_features()),
        "archive": PrunedFeatureStore().read_all(),
    }


@router.get("/feature-importance")
def get_feature_importance() -> dict:
    return {"importance": model_importance()}


@router.get("/prices/{ticker}")
def get_price_history(ticker: str, interval: str = "daily") -> dict:
    try:
        history = intraday_price_history(ticker) if interval == "hourly" else daily_price_history(ticker)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"no {interval} price history for {ticker}")
    return {"ticker": ticker, "interval": interval, "prices": price_series(history)}


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
