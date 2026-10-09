"""Thin JSON serving layer over the existing pipeline -- every endpoint here just
wraps an already-tested pure function from `features/`. No new business logic.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

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
    FeesResponse,
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
    OvernightForecastRequest,
    OvernightForecastResponse,
    OvernightCurrentPriceResponse,
    OvernightModelResponse,
    OvernightActualsRequest,
    OvernightActualsResponse,
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
from stock_picker.paper.book import (
    load_paper_book,
    paper_book_view,
    refresh_paper_book,
    with_current_close_quotes,
)
from stock_picker.training.replay import replay_morning
from stock_picker.features.catalog import (
    compute_formulas_all,
    correlation_matrix,
    coverage_report,
    describe_all,
    experimental_features,
    examples_all,
    list_feature_columns,
    model_derived_features,
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
from stock_picker.features.registry import TICKER_ENTITY, build_registry, experimental_views, model_derived_views
from stock_picker.features.selection import selected_features
from stock_picker.features.stacked_svm import PRODUCTION_MODEL_DERIVED_COLUMNS, RESEARCH_SVM_COLUMNS
from stock_picker.features.trades import (
    peak_working_by_day,
    position_summaries,
    trade_history,
    trade_log,
)
from stock_picker.storage.feature_exclusion_store import DEFAULT_REASON, PrunedFeatureStore
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, MassiveOvernightError, TICKER_PATTERN
from stock_picker.ingestion.session import cash_session_window
from stock_picker.storage.fee_store import FeeStore
from stock_picker.storage.trade_store import ConflictingTradeFeeError, Trade, TradeStore
from stock_picker.storage.ticker_blacklist_store import blacklisted_tickers
from stock_picker.storage.training_config_store import ModelChoice, TrainingConfigStore
from stock_picker.storage.training_run_store import TrainingRunStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training import job as training_job
from stock_picker.training import morning_check as morning_check_job
from stock_picker.training import morning_job as morning_scan_job
from stock_picker.features.earnings import fetch_recent_earnings_tickers
from stock_picker.training.news_day_judge import fetch_recent_news_checks, news_blocks_buy
from stock_picker.training.buy_signal import DEFAULT_THRESHOLD, compute_buy_signals
from stock_picker.training.freshness import pipeline_freshness
from stock_picker.training.morning import load_cached_signals
from stock_picker.training.ensemble import ensemble_composition, selected_model_specs
from stock_picker.training.importance import model_importance
from stock_picker.training.job import JobStatus
from stock_picker.training.main import MODEL_NAME
from stock_picker.training.model import TRAINABLE_MODEL_TYPES
from stock_picker.training.model_registry import describe_model_types
from stock_picker.training.overnight_model import MODEL_NAME as OVERNIGHT_MODEL_NAME
from stock_picker.training.overnight_service import model_summary, serialize_result, serve_overnight_forecast
from stock_picker.training.overnight_hold import observed_next_open

router = APIRouter(prefix="/api")


@router.get("/overnight/model")
def get_overnight_model() -> OvernightModelResponse:
    store = ModelStore()
    model = store.read(OVERNIGHT_MODEL_NAME) if store.exists(OVERNIGHT_MODEL_NAME) else None
    return OvernightModelResponse(**model_summary(model))


@router.get("/overnight/current-price")
def get_overnight_current_price(ticker: str) -> OvernightCurrentPriceResponse:
    """A fresh, timestamped trade for an editable assumed-close prefill."""
    name = ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(name):
        raise HTTPException(status_code=422, detail="invalid ticker")
    now = datetime.now(ZoneInfo("America/New_York"))
    try:
        window = cash_session_window(now)
    except (LookupError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=503, detail="Exchange session calendar is unavailable; enter an assumed close manually") from exc
    if window is None or not window.contains(now):
        raise HTTPException(status_code=503, detail="Current price is available only during regular exchange hours; enter an assumed close manually")
    session = now.date()
    try:
        trade = MassiveOvernightClient().fetch_current_trade(name, session)
    except MassiveOvernightError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not window.contains(datetime.now(ZoneInfo("America/New_York"))) or not window.contains(trade.observed_at):
        raise HTTPException(status_code=503, detail="No regular-session trade is available; enter an assumed close manually")
    if trade.observed_at > trade.fetched_at + timedelta(seconds=5) or trade.fetched_at - trade.observed_at > timedelta(minutes=5):
        raise HTTPException(status_code=503, detail="No trade in the last five minutes; enter an assumed close manually")
    return OvernightCurrentPriceResponse(
        ticker=name, price=trade.price, observed_at=trade.observed_at,
        fetched_at=trade.fetched_at, session_open_at=window.open_at,
        session_close_at=window.close_at, source="massive_last_trade",
    )


@router.post("/overnight/forecast")
def post_overnight_forecast(body: OvernightForecastRequest) -> OvernightForecastResponse:
    ticker = body.ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise HTTPException(status_code=422, detail="invalid ticker")
    session = datetime.now(ZoneInfo("America/New_York")).date()
    try:
        result = serve_overnight_forecast(
            ticker=ticker, session=session, assumed_close=body.assumed_close,
            step=body.step, shares=body.shares,
            exit_today_cost_per_share=body.exit_today_cost_per_share,
            exit_next_open_cost_per_share=body.exit_next_open_cost_per_share,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MassiveOvernightError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return OvernightForecastResponse(**serialize_result(result))


@router.post("/what-if/overnight-actuals")
def post_overnight_actuals(body: OvernightActualsRequest) -> OvernightActualsResponse:
    """On-demand hindsight; never changes the persisted morning paper book."""
    tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in body.tickers))
    if any(not TICKER_PATTERN.fullmatch(ticker) for ticker in tickers):
        raise HTTPException(status_code=422, detail="invalid ticker")
    return OvernightActualsResponse(rows=[
        asdict(observed_next_open(ticker, body.as_of)) for ticker in tickers
    ])


@router.get("/catalog")
def get_catalog() -> CatalogResponse:
    history = sample_history()
    return CatalogResponse(
        catalog=list_feature_columns(history),
        descriptions=describe_all(history),
        formulas=compute_formulas_all(history),
        examples=examples_all(history),
        experimental_features={name: asdict(feature) for name, feature in experimental_features().items()},
        model_derived_features={
            name: {**asdict(feature), "status": "production_eligible"}
            for name, feature in model_derived_features().items()
        },
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
    fill = Trade(
        ticker=trade.ticker,
        side=trade.side,
        shares=trade.shares,
        price=trade.price,
        executed_at=_executed_at(trade.executed_at),
        manual_fee=trade.fee or 0.0,
    )
    try:
        TradeStore().append(fill)
    except ConflictingTradeFeeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return TradesResponse(trades=trade_history(trade_log()))


@router.get("/quotes")
def get_quotes(tickers: str, as_of: str | None = None) -> QuotesResponse:
    """Live by default. If `as_of` names a session that has already settled
    (an earlier date, or today's date once the close has settled), serve the
    real closing bar from PriceStore instead of a live quote -- a live fetch
    returns nothing useful for a past/closed session (no fresh print to
    return), which otherwise leaves Prev/Open/Close blank for any day-old
    scan. Same data source What if already trusts for settled days
    (paper/book.py's session_quotes_from_prices).

    Today's bars can arrive one ticker at a time before the nightly pull.
    Preserve completed bars and fill only missing names from the live quote
    chain; a partial PriceStore result must not blank the rest of the scan.
    Never substitute today's live quote for a missing historical bar.
    """
    names = [ticker for ticker in tickers.split(",") if ticker]
    if as_of:
        from stock_picker.ingestion.session import cash_session_date, session_has_closed
        from stock_picker.paper.book import session_quotes_from_prices

        as_of_date = date.fromisoformat(as_of)
        session_day = cash_session_date()
        if as_of_date < session_day or session_has_closed():
            settled = session_quotes_from_prices(names, as_of_date)
            if as_of_date < session_day:
                return QuotesResponse(quotes=quote_summaries(settled))
            missing = [ticker for ticker in names if ticker not in settled]
            if missing:
                live = fetch_ticker_quotes(missing, as_of=as_of_date)
                settled.update({ticker: live[ticker] for ticker in missing if ticker in live})
            return QuotesResponse(quotes=quote_summaries(settled))
    return QuotesResponse(quotes=quote_summaries(fetch_ticker_quotes(names)))


def _prior_closes(tickers: list[str], as_of: str) -> dict[str, float]:
    """Friday's Close for `as_of` -- same `_prior_close` What if uses."""
    from stock_picker.paper.book import _prior_close
    from stock_picker.storage.price_store import PriceStore

    prices = PriceStore()
    out: dict[str, float] = {}
    for ticker in tickers:
        try:
            history = prices.read(ticker)
        except FileNotFoundError:
            continue
        prev = _prior_close(history, as_of)
        if prev is not None:
            out[ticker] = prev
    return out


def _signal_payload(signal, prev_close: float | None = None) -> dict:
    if isinstance(signal, dict):
        payload = dict(signal)
    else:
        payload = asdict(signal)
    if payload.get("prev_close") is None and prev_close is not None:
        payload["prev_close"] = prev_close
    payload["news_blocks"] = news_blocks_buy(
        payload.get("news_flag"),
        payload.get("open_price"),
        payload.get("prev_close"),
        (payload.get("news_check") or {}).get("status"),
    )
    return payload


def _signals_with_gaps(signals: list, as_of: str) -> list[dict]:
    """Skip/still-buy needs the gap. Fill a blank prev_close from PriceStore."""
    rows = [s if isinstance(s, dict) else asdict(s) for s in signals]
    missing = [row["ticker"] for row in rows if row.get("prev_close") is None]
    filled = _prior_closes(missing, as_of) if missing else {}
    return [_signal_payload(row, filled.get(row["ticker"])) for row in rows]


def _cached_buy_signal(threshold: float, kind: str = "fit") -> BuySignalResponse | None:
    payload = load_cached_signals(kind=kind)
    if payload is None:
        return None
    as_of = payload["as_of"]
    return BuySignalResponse(
        as_of=as_of,
        threshold=payload.get("threshold") if payload.get("threshold") is not None else threshold,
        signals=_signals_with_gaps(payload.get("signals") or [], as_of),
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
    kind=rank loads the rank-model top-K (scores are not percents)."""
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
            news_fetcher=fetch_recent_news_checks,
        )
        return BuySignalResponse(
            as_of=result.as_of,
            threshold=result.threshold,
            signals=_signals_with_gaps(result.signals, result.as_of),
            scored_count=result.scored_count,
            skipped=result.skipped,
            top_drivers=[{"feature": name, "importance": value} for name, value in result.top_drivers],
            cached=False,
        )
    result = compute_buy_signals(
        threshold=threshold,
        earnings_fetcher=fetch_recent_earnings_tickers,
        news_fetcher=fetch_recent_news_checks,
    )
    return BuySignalResponse(
        as_of=result.as_of,
        threshold=result.threshold,
        signals=_signals_with_gaps(result.signals, result.as_of),
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


@router.get("/ticker-blacklist")
def get_ticker_blacklist() -> list[str]:
    """Current scoring exclusions, available for optional historical comparisons."""
    return sorted(blacklisted_tickers())


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
    from stock_picker.ingestion.session import cash_session_date, cash_session_has_ended

    if cash_session_has_ended():
        from stock_picker.ingestion.polygon_client import fetch_polygon_closing_quotes

        today = cash_session_date()
        missing = sorted({
            pick.ticker for pick in picks
            if pick.as_of == today.isoformat() and not pick.scan_id
            and pick.session_return is None
        })
        if missing:
            closes = fetch_polygon_closing_quotes(missing, as_of=today)
            picks = with_current_close_quotes(picks, today, closes)
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


@router.get("/fees")
def get_fees() -> FeesResponse:
    return FeesResponse(fees=[fee.__dict__ for fee in FeeStore().read()])


@router.get("/positions")
def get_positions() -> PositionsResponse:
    trades = trade_log()
    summaries = position_summaries(trades, {})
    open_tickers = sorted(
        {row["ticker"] for row in summaries if not row["closed"] and row["shares"] > 0}
    )
    quotes = fetch_ticker_quotes(open_tickers) if open_tickers else {}
    from stock_picker.ingestion.polygon_client import fetch_polygon_closing_quotes
    from stock_picker.ingestion.session import cash_session_date, cash_session_has_ended

    hold_quotes: dict[str, dict] = {}
    hold_through = None
    if cash_session_has_ended():
        hold_through = cash_session_date()
        today = hold_through.isoformat()
        closed_today = sorted(
            {row["ticker"] for row in summaries if row.get("day") == today}
        )
        if closed_today:
            hold_quotes = fetch_polygon_closing_quotes(closed_today, as_of=hold_through)
    positions = apply_hold_to_close(
        summaries if not quotes else position_summaries(trades, quotes),
        completed_through=hold_through,
        live_quotes=hold_quotes or None,
    )
    return PositionsResponse(
        positions=positions,
        peak_working=peak_working_by_day(trades),
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
    selected_run_id = TrainingConfigStore().read().selected_run_id
    model_name = f"{MODEL_NAME}_{selected_run_id}" if selected_run_id else MODEL_NAME
    importance = model_importance(
        model_name, store=ModelStore(), include_diagnostic=selected_run_id is None
    )
    return ImportanceResponse(importance=importance["blended"], by_model_type=importance["by_model_type"])


@router.get("/model-info")
def get_model_info() -> ModelInfoResponse:
    store = ModelStore()
    selected_run_id = TrainingConfigStore().read().selected_run_id
    model_name = f"{MODEL_NAME}_{selected_run_id}" if selected_run_id else MODEL_NAME
    if not store.exists(model_name):
        return ModelInfoResponse(models=[])
    return ModelInfoResponse(models=ensemble_composition(store.read(model_name)))


@router.get("/model-types")
def get_model_types() -> ModelTypesResponse:
    return ModelTypesResponse(model_types=[ModelTypeInfoModel(**asdict(info)) for info in describe_model_types()])


@router.get("/feature-selection")
def get_feature_selection() -> FeatureSelectionResponse:
    features = selected_features()
    return FeatureSelectionResponse(included_features=sorted(features) if features is not None else None)


@router.post("/feature-selection")
def set_feature_selection(body: FeatureSelectionRequest) -> FeatureSelectionResponse:
    included = set(body.included_features)
    experimental = sorted(included & set(RESEARCH_SVM_COLUMNS))
    if experimental:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"SVM-derived research outputs cannot be selected for production training: {', '.join(experimental)}",
        )
    raw_catalog = list_feature_columns(sample_history())
    raw_features = {name for columns in raw_catalog.values() for name in columns}
    unknown = sorted(included - raw_features - set(PRODUCTION_MODEL_DERIVED_COLUMNS))
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown feature names cannot be selected for production training: {', '.join(unknown)}",
        )
    usable_raw = (included & raw_features) - pruned_features()
    if not usable_raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Feature selection requires at least one unpruned raw pipeline feature alongside any model-derived feature.",
        )
    TrainingConfigStore().write_included_features(included)
    return FeatureSelectionResponse(included_features=sorted(included))


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


def _morning_scan_status() -> MorningScanStatus:
    from stock_picker.training.morning import morning_lock_held

    state = morning_scan_job.status()
    if state.status != "running" and morning_lock_held():
        return MorningScanStatus(status="running")
    return MorningScanStatus(
        status=state.status,
        started_at=state.started_at,
        completed_at=state.completed_at,
        error=state.error,
    )


@router.post("/morning-scan")
def start_morning_scan() -> MorningScanStatus:
    from stock_picker.training.morning import morning_lock_held

    if morning_scan_job.status().status == "running" or morning_lock_held():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a morning scan is already in progress"
        )
    started = morning_scan_job.start()
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="a morning scan is already in progress"
        )
    return _morning_scan_status()


@router.get("/morning-scan")
def get_morning_scan() -> MorningScanStatus:
    return _morning_scan_status()


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
        experimental_views=[asdict(view) for view in experimental_views()],
        model_derived_views=[asdict(view) for view in model_derived_views()],
    )
