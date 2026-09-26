"""Thin JSON serving layer over the existing pipeline -- every endpoint here just
wraps an already-tested pure function from `features/`. No new business logic.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, status

from stock_picker.api.models import (
    BenchmarkHold,
    BenchmarkReturnsResponse,
    BuySignalResponse,
    CatalogResponse,
    CorrelationResponse,
    CoverageResponse,
    FeatureSelectionRequest,
    FeatureSelectionResponse,
    FeatureValuesResponse,
    ImportanceResponse,
    LiveModelResponse,
    MorningCheckRequest,
    MorningCheckResponse,
    MorningJobSettings,
    MorningScanStatus,
    ModelInfoResponse,
    PipelineFreshnessResponse,
    ModelSelectionRequest,
    ModelSelectionResponse,
    ModelTypesResponse,
    PaperBookResponse,
    PaperReplayRequest,
    PositionsResponse,
    PriceHistoryResponse,
    PruneRequest,
    PrunedFeaturesResponse,
    QuotesResponse,
    RegistryResponse,
    SetLiveModelRequest,
    TradeCreate,
    TradesResponse,
    TrainingRunsResponse,
    UniverseResponse,
)
from stock_picker.api.models import ModelChoice as ModelChoiceModel
from stock_picker.api.models import ModelTypeInfo as ModelTypeInfoModel
from stock_picker.api.models import TrainingRunRecord as TrainingRunRecordModel
from stock_picker.features.benchmark import (
    fetch_benchmark_hold,
    fetch_benchmark_overnight,
    fetch_benchmark_returns,
)
from stock_picker.features.hold_to_close import apply_hold_to_close
from stock_picker.paper.book import load_paper_book, paper_book_view, refresh_paper_book
from stock_picker.training.replay import replay_morning
from stock_picker.features.catalog import (
    compute_formulas_all,
    correlation_matrix,
    coverage_report,
    describe_all,
    examples_all,
    list_feature_columns,
    top_correlated_pairs,
)
from stock_picker.features.catalog_loader import STATS_SAMPLE_SIZE, feature_tables, sample_history
from stock_picker.features.price_history import (
    daily_price_history,
    feature_value_rows,
    intraday_price_history,
    price_series,
)
from stock_picker.features.pruning import pruned_features
from stock_picker.features.quotes import fetch_ticker_quotes, quote_summaries
from stock_picker.features.registry import TICKER_ENTITY, build_registry
from stock_picker.features.selection import selected_features
from stock_picker.features.trades import (
    position_summaries,
    time_weighted_working_by_day,
    trade_history,
    trade_log,
)
from stock_picker.storage.feature_exclusion_store import DEFAULT_REASON, PrunedFeatureStore
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.trade_store import Trade, TradeStore
from stock_picker.storage.training_config_store import ModelChoice, TrainingConfigStore
from stock_picker.storage.training_run_store import TrainingRunStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training import job as training_job
from stock_picker.training import morning_check as morning_check_job
from stock_picker.training import morning_job as morning_scan_job
from stock_picker.features.earnings import fetch_recent_earnings_tickers
from stock_picker.training.news_day_judge import fetch_recent_news_flags, news_blocks_buy
from stock_picker.training.buy_signal import DEFAULT_THRESHOLD, compute_buy_signals
from stock_picker.training.freshness import pipeline_freshness
from stock_picker.training.morning import load_cached_signals
from stock_picker.training.ensemble import ensemble_composition, selected_model_specs
from stock_picker.training.importance import model_importance
from stock_picker.training.job import JobStatus
from stock_picker.training.main import MODEL_NAME
from stock_picker.training.model import TRAINABLE_MODEL_TYPES
from stock_picker.training.model_registry import describe_model_types

router = APIRouter(prefix="/api")


@router.get("/catalog")
def get_catalog() -> CatalogResponse:
    history = sample_history()
    return CatalogResponse(
        catalog=list_feature_columns(history),
        descriptions=describe_all(history),
        formulas=compute_formulas_all(history),
        examples=examples_all(history),
    )


@router.get("/coverage")
def get_coverage() -> CoverageResponse:
    report = coverage_report(feature_tables(limit=STATS_SAMPLE_SIZE))
    return CoverageResponse(coverage=report["non_null_pct"].to_dict())


@router.get("/correlation")
def get_correlation() -> CorrelationResponse:
    corr = correlation_matrix(feature_tables(limit=STATS_SAMPLE_SIZE))
    return CorrelationResponse(
        columns=list(corr.columns),
        matrix=corr.where(corr.notna(), None).values.tolist(),
        top_pairs=top_correlated_pairs(corr),
    )


@router.get("/trades")
def get_trades() -> TradesResponse:
    return TradesResponse(trades=trade_history(trade_log()))


def _executed_at(value: str | None) -> str:
    if not value:
        return datetime.now().astimezone().isoformat()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="executed_at must be ISO 8601"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.isoformat()


@router.post("/trades")
def create_trade(trade: TradeCreate) -> TradesResponse:
    TradeStore().append(
        Trade(
            ticker=trade.ticker,
            side=trade.side,
            shares=trade.shares,
            price=trade.price,
            executed_at=_executed_at(trade.executed_at),
        )
    )
    return TradesResponse(trades=trade_history(trade_log()))


@router.get("/quotes")
def get_quotes(tickers: str) -> QuotesResponse:
    return QuotesResponse(quotes=quote_summaries(fetch_ticker_quotes(tickers.split(","))))


def _signal_payload(signal) -> dict:
    if isinstance(signal, dict):
        payload = dict(signal)
    else:
        payload = asdict(signal)
    payload["news_blocks"] = news_blocks_buy(
        payload.get("news_flag"),
        payload.get("open_price"),
        payload.get("prev_close"),
    )
    return payload


def _cached_buy_signal(threshold: float, kind: str = "fit") -> BuySignalResponse | None:
    payload = load_cached_signals(kind=kind)
    if payload is None:
        return None
    return BuySignalResponse(
        as_of=payload["as_of"],
        threshold=payload.get("threshold") if payload.get("threshold") is not None else threshold,
        signals=[_signal_payload(s) for s in (payload.get("signals") or [])],
        scored_count=payload.get("scored_count", 0),
        skipped=payload.get("skipped") or [],
        top_drivers=payload.get("top_drivers") or [],
        cached=True,
    )


def _empty_buy_signal(threshold: float) -> BuySignalResponse:
    from stock_picker.ingestion.session import cash_session_date

    return BuySignalResponse(
        as_of=cash_session_date().isoformat(),
        threshold=threshold,
        signals=[],
        scored_count=0,
        skipped=[],
        top_drivers=[],
        cached=True,
    )


@router.get("/buy-signal")
def get_buy_signal(
    threshold: float = DEFAULT_THRESHOLD, live: bool = False, kind: str = "fit"
) -> BuySignalResponse:
    """Prefer this morning's saved scan. `live=true` forces a full rescore.
    kind=rank loads lambdarank top-K (scores are not percents)."""
    if not live:
        cached = _cached_buy_signal(threshold, kind=kind)
        if cached is not None:
            return cached
        from stock_picker.ingestion.session import session_has_closed

        if not session_has_closed():
            return _empty_buy_signal(threshold)
    if kind == "rank":
        from stock_picker.training.buy_signal import compute_rank_signals
        from stock_picker.training.rank_model import RANK_TOP_K

        result = compute_rank_signals(
            top_k=RANK_TOP_K,
            earnings_fetcher=fetch_recent_earnings_tickers,
            news_fetcher=fetch_recent_news_flags,
        )
        return BuySignalResponse(
            as_of=result.as_of,
            threshold=result.threshold,
            signals=[_signal_payload(signal) for signal in result.signals],
            scored_count=result.scored_count,
            skipped=result.skipped,
            top_drivers=[{"feature": name, "importance": value} for name, value in result.top_drivers],
            cached=False,
        )
    result = compute_buy_signals(
        threshold=threshold,
        earnings_fetcher=fetch_recent_earnings_tickers,
        news_fetcher=fetch_recent_news_flags,
    )
    return BuySignalResponse(
        as_of=result.as_of,
        threshold=result.threshold,
        signals=[_signal_payload(signal) for signal in result.signals],
        scored_count=result.scored_count,
        skipped=result.skipped,
        top_drivers=[{"feature": name, "importance": value} for name, value in result.top_drivers],
        cached=False,
    )


@router.get("/universe")
def get_universe() -> UniverseResponse:
    # Deliberately cheap (just a parquet read, no live quotes) so the UI can
    # show "scanning N tickers" up front, before the user ever clicks
    # "check this morning's prices" -- compute_buy_signals itself only
    # reveals this count as a side effect of the full, slower live scan.
    return UniverseResponse(active_ticker_count=len(UniverseStore().active_tickers()))


@router.get("/benchmark-returns")
def get_benchmark_returns(dates: str) -> BenchmarkReturnsResponse:
    requested = [d for d in dates.split(",") if d]
    hold_raw = fetch_benchmark_hold(min(requested), max(requested)) if requested else None
    hold = (
        BenchmarkHold(start=hold_raw["from"], end=hold_raw["to"], pct=hold_raw["return"])
        if hold_raw
        else None
    )
    return BenchmarkReturnsResponse(
        returns=fetch_benchmark_returns(requested),
        overnight=fetch_benchmark_overnight(requested),
        hold=hold,
    )


@router.get("/paper-book")
def get_paper_book(
    kind: str = "both",
    top_k: int | None = None,
    fit_top_k: int | None = None,
    rank_top_k: int | None = None,
) -> PaperBookResponse:
    if kind not in ("fit", "rank", "both"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="kind must be fit, rank, or both"
        )
    picks = load_paper_book()
    return PaperBookResponse(
        **paper_book_view(
            picks,
            kind=kind,
            top_k=top_k,
            fit_top_k=fit_top_k,
            rank_top_k=rank_top_k,
        )
    )


@router.post("/paper-book/rebuild")
def post_paper_book_rebuild() -> PaperBookResponse:
    picks = refresh_paper_book()
    return PaperBookResponse(**paper_book_view(picks, kind="both"))


@router.post("/paper-book/replay")
def post_paper_book_replay(body: PaperReplayRequest) -> PaperBookResponse:
    as_of = date.fromisoformat(body.as_of)
    replay_morning(as_of, model_run_id=body.model_run_id, threshold=body.threshold)
    return PaperBookResponse(**paper_book_view(load_paper_book(), kind="both"))


@router.get("/positions")
def get_positions() -> PositionsResponse:
    trades = trade_log()
    summaries = position_summaries(trades, {})
    open_tickers = sorted(
        {row["ticker"] for row in summaries if not row["closed"] and row["shares"] > 0}
    )
    quotes = fetch_ticker_quotes(open_tickers) if open_tickers else {}
    positions = apply_hold_to_close(
        summaries if not quotes else position_summaries(trades, quotes)
    )
    return PositionsResponse(
        positions=positions,
        peak_working=time_weighted_working_by_day(trades),
    )


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


@router.get("/model-types")
def get_model_types() -> ModelTypesResponse:
    return ModelTypesResponse(model_types=[ModelTypeInfoModel(**asdict(info)) for info in describe_model_types()])


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
        available_model_types=TRAINABLE_MODEL_TYPES,
    )


@router.post("/model-selection")
def set_model_selection(body: ModelSelectionRequest) -> ModelSelectionResponse:
    choices = [ModelChoice(model_type=c.model_type, weight=c.weight) for c in body.model_choices]
    TrainingConfigStore().write_model_choices(choices)
    return ModelSelectionResponse(
        model_choices=[ModelChoiceModel(model_type=c.model_type, weight=c.weight) for c in choices],
        available_model_types=TRAINABLE_MODEL_TYPES,
    )


@router.delete("/model-selection")
def clear_model_selection() -> ModelSelectionResponse:
    TrainingConfigStore().write_model_choices(None)
    return ModelSelectionResponse(model_choices=None, available_model_types=TRAINABLE_MODEL_TYPES)


@router.post("/morning-check")
def start_morning_check(body: MorningCheckRequest | None = None) -> MorningCheckResponse:
    which = (body.which if body is not None else "both")
    started = morning_check_job.start(which=which)
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a morning check is already in progress"
        )
    state = morning_check_job.status()
    return MorningCheckResponse(**asdict(state))


@router.get("/morning-check")
def get_morning_check() -> MorningCheckResponse:
    return MorningCheckResponse(**asdict(morning_check_job.status()))


@router.get("/morning-check/runs")
def list_morning_checks() -> list[dict]:
    return morning_check_job.saved_runs()


@router.post("/morning-check/load")
def load_morning_check(body: MorningCheckRequest) -> MorningCheckResponse:
    if not body.run_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="run_id required")
    loaded = morning_check_job.load_saved(body.run_id)
    if loaded is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="test run not found")
    return MorningCheckResponse(**asdict(loaded))


@router.get("/morning-job")
def get_morning_job() -> MorningJobSettings:
    return MorningJobSettings(enabled=TrainingConfigStore().read().morning_job_enabled)


@router.put("/morning-job")
def set_morning_job(body: MorningJobSettings) -> MorningJobSettings:
    TrainingConfigStore().write_morning_job_enabled(body.enabled)
    return MorningJobSettings(enabled=body.enabled)


@router.post("/morning-scan")
def start_morning_scan() -> MorningScanStatus:
    started = morning_scan_job.start()
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a morning scan is already in progress"
        )
    state = morning_scan_job.status()
    return MorningScanStatus(
        status=state.status,
        started_at=state.started_at,
        completed_at=state.completed_at,
        error=state.error,
    )


@router.get("/morning-scan")
def get_morning_scan() -> MorningScanStatus:
    state = morning_scan_job.status()
    return MorningScanStatus(
        status=state.status,
        started_at=state.started_at,
        completed_at=state.completed_at,
        error=state.error,
    )


@router.post("/training/run")
def start_training_run() -> JobStatus:
    started = training_job.start(included_features=selected_features(), model_specs=selected_model_specs())
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a training run is already in progress"
        )
    return training_job.status()


@router.get("/training/status")
def get_training_status() -> JobStatus:
    return training_job.status()


@router.get("/training/runs")
def get_training_runs() -> TrainingRunsResponse:
    model_store = ModelStore()
    return TrainingRunsResponse(
        runs=[
            TrainingRunRecordModel(
                **asdict(record), has_archived_model=model_store.exists(f"{MODEL_NAME}_{record.run_id}")
            )
            for record in TrainingRunStore().read_all()
        ]
    )


@router.get("/live-model")
def get_live_model() -> LiveModelResponse:
    selected_run_id = TrainingConfigStore().read().selected_run_id
    live_run_id = selected_run_id
    if live_run_id is None:
        completed = [r for r in TrainingRunStore().read_all() if r.status == "completed"]
        if completed:
            live_run_id = completed[0].run_id  # newest first, see read_all()
    return LiveModelResponse(selected_run_id=selected_run_id, live_run_id=live_run_id)


@router.post("/live-model")
def set_live_model(body: SetLiveModelRequest) -> LiveModelResponse:
    if not ModelStore().exists(f"{MODEL_NAME}_{body.run_id}"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run {body.run_id} has no archived model to select",
        )
    TrainingConfigStore().write_selected_run_id(body.run_id)
    return get_live_model()


@router.delete("/live-model")
def clear_live_model() -> LiveModelResponse:
    TrainingConfigStore().write_selected_run_id(None)
    return get_live_model()


@router.get("/pipeline-freshness")
def get_pipeline_freshness() -> PipelineFreshnessResponse:
    return PipelineFreshnessResponse(**asdict(pipeline_freshness()))


@router.get("/prices/{ticker}")
def get_price_history(ticker: str, interval: Literal["daily", "hourly"] = "daily") -> PriceHistoryResponse:
    try:
        history = intraday_price_history(ticker) if interval == "hourly" else daily_price_history(ticker)
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no {interval} price history for {ticker}",
        )
    return PriceHistoryResponse(ticker=ticker, interval=interval, prices=price_series(history))


@router.get("/features/{ticker}")
def get_feature_values(ticker: str) -> FeatureValuesResponse:
    try:
        features = FeatureStore().read(ticker)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no feature data for {ticker}"
        )
    return FeatureValuesResponse(
        ticker=ticker, columns=list(features.columns), rows=feature_value_rows(features)
    )


@router.get("/registry")
def get_registry() -> RegistryResponse:
    feature_views, feature_services = build_registry(sample_history())
    return RegistryResponse(
        entities=[asdict(TICKER_ENTITY)],
        feature_views=[asdict(view) for view in feature_views],
        feature_services=[asdict(service) for service in feature_services],
    )
