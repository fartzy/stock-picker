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
    # ISO 8601 with offset. None = stamp as now (the old form behavior).
    executed_at: str | None = None


class PruneRequest(BaseModel):
    reason: str | None = None


class FeatureSelectionRequest(BaseModel):
    included_features: list[str]


class ModelChoice(BaseModel):
    model_type: str
    weight: float = 1.0


class ModelSelectionRequest(BaseModel):
    model_choices: list[ModelChoice]


class SetLiveModelRequest(BaseModel):
    run_id: str


# ---- responses ----


class CatalogResponse(BaseModel):
    catalog: dict[str, list[str]]
    descriptions: dict[str, str]
    formulas: dict[str, str]
    examples: dict[str, str]


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
    # 8:40 AM CT Open -> 2:55 PM CT Close on the buy's session. None if
    # the session is still in progress or PriceStore has no bar that day.
    hold_open_price: float | None = None
    hold_close_price: float | None = None
    hold_close_pnl: float | None = None


class PositionsResponse(BaseModel):
    positions: list[Position]
    # NY session date -> dollars on the book while anything is on (not peak,
    # not a 6.5h smear of the empty afternoon).
    peak_working: dict[str, float] = {}


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
    # Which model types the Models picker offers (TRAINABLE_MODEL_TYPES).
    # logistic_regression is excluded (diagnostic). lightgbm_rank is included
    # but trained as a parallel pickle, not blended with the return models.
    available_model_types: list[str]


class ImportanceResponse(BaseModel):
    importance: dict[str, float]
    by_model_type: dict[str, dict[str, float]]


class ModelInfoResponse(BaseModel):
    models: list[EnsembleMemberInfo]


class ModelTypeInfo(BaseModel):
    model_type: str
    display_name: str
    category: str
    package: str
    package_version: str
    source_file: str
    source_line: int
    # None if the origin remote can't be resolved -- the UI falls back to
    # showing source_file:source_line as plain, non-linked text.
    github_url: str | None
    description: str


class ModelTypesResponse(BaseModel):
    model_types: list[ModelTypeInfo]


class TrainingRunRecord(BaseModel):
    run_id: str
    status: Literal["completed", "failed"]
    started_at: str
    completed_at: str
    duration_seconds: float
    git_commit: str | None = None
    # None on a run that failed before this provenance was known -- see
    # training/main.py's run_training().
    train_tickers: list[str] | None = None
    holdout_tickers: list[str] | None = None
    date_range: tuple[str, str] | None = None
    resolved_features: list[str] | None = None
    model_specs: list[dict] | None = None
    fold_metrics: list[dict] | None = None
    holdout_metrics: dict | None = None
    threshold_sweep: list[dict] | None = None
    error: str | None = None
    # Computed, not persisted on the storage-layer record -- whether this
    # run's model was archived under its own run_id (see training/main.py's
    # run_training()). False for every run before that archival feature
    # existed, and for any run whose model_specs archival failed for some
    # other reason.
    has_archived_model: bool = False


class TrainingRunsResponse(BaseModel):
    # Newest first -- see storage/training_run_store.py's read_all().
    runs: list[TrainingRunRecord]


class LiveModelResponse(BaseModel):
    # None = no explicit choice, "latest" is live.
    selected_run_id: str | None
    # The resolved answer -- selected_run_id if set, else the most recent
    # completed run's id. None only if there's no completed run at all yet.
    live_run_id: str | None


class PipelineFreshnessResponse(BaseModel):
    as_of: str
    last_completed_session: str
    feature_snapshot_date: str | None
    features_ok: bool
    feature_age_weekdays: int | None
    model_trained_through: str | None
    model_ok: bool
    model_age_weekdays: int | None
    ready_for_inference: bool
    live_run_id: str | None
    detail: str


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


class FeatureValuesResponse(BaseModel):
    ticker: str
    columns: list[str]
    # Each row: {"date": iso string, <feature>: value | null, ...} -- flat
    # rather than a nested per-row dict, since the frontend just indexes a
    # row by column name the same way regardless of which key it's reading.
    rows: list[dict[str, float | str | None]]


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


class BuySignalRow(BaseModel):
    ticker: str
    predicted_return: float
    open_price: float
    snapshot_date: str
    # Headline from Finnhub company-news on this name only, if it looks
    # like a trial hold / FDA / dilution day. None = no flag.
    news_flag: str | None = None
    prev_close: float | None = None
    news_blocks: bool = False


class SkippedTicker(BaseModel):
    ticker: str
    reason: str


class TopDriver(BaseModel):
    feature: str
    importance: float


class BuySignalResponse(BaseModel):
    as_of: str
    threshold: float
    signals: list[BuySignalRow]
    scored_count: int
    skipped: list[SkippedTicker]
    top_drivers: list[TopDriver]
    # True when this is the 8:31 job's saved scan, not a live rescore.
    cached: bool = False


class UniverseResponse(BaseModel):
    active_ticker_count: int


class BenchmarkHold(BaseModel):
    start: str
    end: str
    pct: float


class PaperPickRow(BaseModel):
    rank: int
    ticker: str
    predicted: float | None = None
    open_price: float | None = None
    close_price: float | None = None
    session_return: float | None = None
    news_flag: str | None = None
    news_blocks: bool = False


class PaperListStats(BaseModel):
    n: int = 0
    n_scored: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    hit_rate: float | None = None
    avg: float | None = None
    n_avoid: int = 0
    avg_ex_news: float | None = None


class PaperBookDay(BaseModel):
    as_of: str
    scan_id: str = ""
    model_run_id: str | None = None
    fit: list[PaperPickRow] = []
    rank: list[PaperPickRow] = []
    fit_avg: float | None = None
    rank_avg: float | None = None
    fit_stats: PaperListStats = PaperListStats()
    rank_stats: PaperListStats = PaperListStats()


class PaperBookResponse(BaseModel):
    days: list[PaperBookDay]
    fit_compound: float | None = None
    rank_compound: float | None = None
    fit_days: int = 0
    rank_days: int = 0
    fit_stats: PaperListStats = PaperListStats()
    rank_stats: PaperListStats = PaperListStats()
    kind: str = "both"
    top_k: int | None = None
    fit_top_k: int | None = None
    rank_top_k: int | None = None
    n_picks: int = 0


class PaperReplayRequest(BaseModel):
    as_of: str
    model_run_id: str | None = None
    threshold: float = 0.005


class FakeQuote(BaseModel):
    ticker: str
    fake_open: float
    last_close: float


class TimedPass(BaseModel):
    which: str
    seconds: float
    scored_count: int
    n_picks: int
    picks: list[dict] = []
    skipped_count: int = 0


class MorningCheckResponse(BaseModel):
    status: str
    which: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    n_quotes: int = 0
    quote_seconds: float | None = None
    quotes: list[FakeQuote] = []
    passes: list[TimedPass] = []


class MorningCheckRequest(BaseModel):
    which: Literal["rank", "fit", "both"] = "both"
    run_id: str | None = None


class MorningCheckRunSummary(BaseModel):
    id: str
    started_at: str | None = None
    completed_at: str | None = None
    which: str | None = None
    status: str | None = None
    n_quotes: int | None = None


class MorningJobSettings(BaseModel):
    enabled: bool


class MorningScanStatus(BaseModel):
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class BenchmarkReturnsResponse(BaseModel):
    # date -> SPY open->close (intraday / cash session). Missing dates omitted.
    returns: dict[str, float]
    # date -> SPY prior close->this close (held overnight, did not sell).
    overnight: dict[str, float] = {}
    # Close-to-close if parked in SPY from the earliest requested date
    # through the latest (not an average of session days).
    hold: BenchmarkHold | None = None
