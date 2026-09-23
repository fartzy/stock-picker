"""Live "what should I buy this morning" scoring -- the app's whole premise in
one function: score every active ticker's current-morning quote through the
persisted ensemble and report which ones clear a confidence threshold, and why.

Open-known rows are built once (`live_rows.prepare_live_rows`) in 200-name
process buckets. Rank and Fit only predict on that matrix -- they do not
each rebuild parquet + seasonality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from stock_picker.features.quotes import fetch_ticker_quotes
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble, predict_ensemble
from stock_picker.training.importance import ensemble_importance
from stock_picker.training.live_rows import LiveRow, prepare_live_rows
from stock_picker.training.main import MODEL_NAME
from stock_picker.training.rank_model import RANK_MODEL_NAME, RANK_TOP_K

# Matches the Models tab's own established default threshold (see
# training/backtest.py's sweep_thresholds and TrainingPanel.tsx).
DEFAULT_THRESHOLD = 0.005

# How many of the model's top blended-importance features to surface as
# general "why" context -- a model-level property, not a per-ticker
# explanation (real per-stock attribution is future work).
TOP_DRIVER_COUNT = 3

# Used for the one global skip entry when no model is persisted yet.
# Deliberately not a valid ticker shape (real symbols are 1-5 uppercase
# letters) -- an earlier version of this used "ALL", which collided with
# Allstate Corporation's real ticker symbol and made every stale-snapshot
# day misreport as "no model trained yet."
NO_MODEL_SENTINEL = ""

# Same 200 / 10 as Yahoo quote batches -- one universe, one bucket size.
SCORE_BUCKET = 200
SCORE_WORKERS = 10


@dataclass
class BuySignal:
    ticker: str
    predicted_return: float
    open_price: float
    snapshot_date: str
    news_flag: str | None = None
    prev_close: float | None = None


@dataclass
class BuySignalResult:
    as_of: str
    threshold: float
    signals: list[BuySignal] = field(default_factory=list)
    scored_count: int = 0
    skipped: list[dict] = field(default_factory=list)
    top_drivers: list[tuple[str, float]] = field(default_factory=list)


def _signals_from_live_rows(
    live_rows: list[LiveRow],
    ensemble: Ensemble,
    threshold: float,
) -> tuple[list[BuySignal], list[dict], int]:
    """Predict on already-built rows. Rank and Fit both call this."""
    skipped: list[dict] = []
    ready: list[LiveRow] = []
    for item in live_rows:
        if item.skipped is not None:
            skipped.append(item.skipped)
            continue
        if item.row is None:
            continue
        ready.append(item)
    if not ready:
        return [], skipped, 0
    frame = pd.concat([item.row for item in ready], ignore_index=True)
    predicted = predict_ensemble(ensemble, frame)
    signals: list[BuySignal] = []
    for item, predicted_return in zip(ready, predicted):
        predicted_return = float(predicted_return)
        if predicted_return > threshold:
            signals.append(
                BuySignal(
                    ticker=item.ticker,
                    predicted_return=predicted_return,
                    open_price=item.open_price,
                    snapshot_date=item.snapshot_date,
                    prev_close=item.prev_close,
                )
            )
    return signals, skipped, len(ready)



def compute_buy_signals(
    threshold: float = DEFAULT_THRESHOLD,
    as_of: date | None = None,
    universe_store: UniverseStore | None = None,
    feature_store: FeatureStore | None = None,
    model_store: ModelStore | None = None,
    config_store: TrainingConfigStore | None = None,
    price_store: PriceStore | None = None,
    quote_fetcher: Callable[[list[str]], dict[str, dict]] = fetch_ticker_quotes,
    earnings_fetcher: Callable[[list[str], date], set[str]] | None = None,
    news_fetcher: Callable[[list[str], date], dict[str, str]] | None = None,
    model_name: str | None = None,
    top_k: int | None = None,
    live_rows: list[LiveRow] | None = None,
) -> BuySignalResult:
    as_of = as_of or date.today()
    universe_store = universe_store or UniverseStore()
    feature_store = feature_store or FeatureStore()
    model_store = model_store or ModelStore()
    config_store = config_store or TrainingConfigStore()
    price_store = price_store or PriceStore()

    # None (the default) means "no explicit choice, use whatever's latest" --
    # that name always gets overwritten by every run regardless of run_id.
    # A selected_run_id instead picks up that specific run's archived copy
    # (see training/main.py's run_training()). /api/live-model validates a
    # selection actually has an archive before it can be set, so this should
    # never need a runtime fallback -- but if a selection somehow predates
    # the archival feature, err toward "no model" rather than silently
    # substituting a different model than the one explicitly chosen.
    selected_run_id = None
    if model_name is None:
        selected_run_id = config_store.read().selected_run_id
        model_name = f"{MODEL_NAME}_{selected_run_id}" if selected_run_id else MODEL_NAME

    if not model_store.exists(model_name):
        reason = (
            "no trained model persisted yet"
            if not selected_run_id
            else f"selected run {selected_run_id} has no archived model"
        )
        return BuySignalResult(
            as_of=as_of.isoformat(),
            threshold=threshold,
            skipped=[{"ticker": NO_MODEL_SENTINEL, "reason": reason}],
        )

    ensemble = model_store.read(model_name)
    tickers = universe_store.active_tickers()
    try:
        quotes = quote_fetcher(tickers, as_of=as_of)
    except TypeError:
        quotes = quote_fetcher(tickers)
    spy_quote = quotes.get("SPY")
    if spy_quote is None:
        try:
            spy_only = quote_fetcher(["SPY"], as_of=as_of)
        except TypeError:
            spy_only = quote_fetcher(["SPY"])
        spy_quote = spy_only.get("SPY")
    spy_open = spy_quote.get("open") if spy_quote else None
    spy_prev_close = spy_quote.get("prev_close") if spy_quote else None

    earnings: set[str] = set()
    if earnings_fetcher is not None:
        try:
            earnings = earnings_fetcher(tickers, as_of) or set()
        except TypeError:
            earnings = set()

    if live_rows is None:
        live_rows = prepare_live_rows(
            tickers=tickers,
            quotes=quotes,
            earnings=earnings,
            feature_store=feature_store,
            price_store=price_store,
            as_of=as_of,
            spy_open=spy_open,
            spy_prev_close=spy_prev_close,
            bucket_size=SCORE_BUCKET,
            workers=SCORE_WORKERS,
        )
    signals, skipped, scored_count = _signals_from_live_rows(live_rows, ensemble, threshold)

    signals.sort(key=lambda signal: signal.predicted_return, reverse=True)
    if top_k is not None:
        signals = signals[:top_k]
    # Tests pass news_fetcher=None (no network). Morning / live API pass
    # Finnhub so only the recommended names get a company-news lookup.
    if news_fetcher is not None and signals:
        try:
            flags = news_fetcher([signal.ticker for signal in signals], as_of) or {}
        except TypeError:
            flags = {}
        for signal in signals:
            signal.news_flag = flags.get(signal.ticker)
    blended_importance = sorted(ensemble_importance(ensemble).items(), key=lambda item: item[1], reverse=True)
    top_drivers = blended_importance[:TOP_DRIVER_COUNT]

    return BuySignalResult(
        as_of=as_of.isoformat(),
        threshold=threshold,
        signals=signals,
        scored_count=scored_count,
        skipped=skipped,
        top_drivers=top_drivers,
    )


def compute_rank_signals(
    top_k: int = RANK_TOP_K,
    as_of: date | None = None,
    **kwargs,
) -> BuySignalResult:
    """Lambdarank top-K. Score is relative rank, not a percent."""
    model_store = kwargs.get("model_store") or ModelStore()
    as_of = as_of or date.today()
    if not model_store.exists(RANK_MODEL_NAME):
        return BuySignalResult(
            as_of=as_of.isoformat(),
            threshold=0.0,
            skipped=[{"ticker": NO_MODEL_SENTINEL, "reason": "no lambdarank model persisted yet"}],
        )
    return compute_buy_signals(
        threshold=float("-inf"),
        as_of=as_of,
        model_name=RANK_MODEL_NAME,
        top_k=top_k,
        **kwargs,
    )
