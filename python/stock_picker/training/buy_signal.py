"""Live "what should I buy this morning" scoring -- the app's whole premise in
one function: score every active ticker's current-morning quote through the
persisted ensemble and report which ones clear a confidence threshold, and why.

Every building block here already exists and is independently tested --
`inference.py` builds the lookahead-safe row and guards against stale/implausible
data, `quotes.py` batch-fetches live quotes in a single call, `importance.py`
already computes the model's blended feature importance. This module is just
the per-ticker loop that ties them together, skipping (and recording why)
rather than crashing on any one ticker's bad data -- one bad quote or a stale
snapshot for GOOG shouldn't prevent scoring the other 499 tickers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from stock_picker.features.quotes import fetch_ticker_quotes
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.importance import ensemble_importance
from stock_picker.training.inference import (
    StaleFeatureSnapshotError,
    build_inference_row,
    predict_signal,
)
from stock_picker.training.main import MODEL_NAME

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


@dataclass
class BuySignal:
    ticker: str
    predicted_return: float
    open_price: float
    snapshot_date: str


@dataclass
class BuySignalResult:
    as_of: str
    threshold: float
    signals: list[BuySignal] = field(default_factory=list)
    scored_count: int = 0
    skipped: list[dict] = field(default_factory=list)
    top_drivers: list[tuple[str, float]] = field(default_factory=list)


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

    signals: list[BuySignal] = []
    skipped: list[dict] = []
    scored_count = 0
    for ticker in tickers:
        if ticker in earnings:
            skipped.append(
                {
                    "ticker": ticker,
                    "reason": "earnings on or since the prior session -- news day, not a pattern day",
                }
            )
            continue
        quote = quotes.get(ticker)
        if quote is None:
            skipped.append({"ticker": ticker, "reason": "no live quote available"})
            continue
        prev_close = quote.get("prev_close")
        if prev_close is None:
            skipped.append({"ticker": ticker, "reason": "no previous close available"})
            continue

        try:
            prior_features = feature_store.read(ticker)
        except FileNotFoundError:
            skipped.append({"ticker": ticker, "reason": "no feature history"})
            continue

        snapshot_date = prior_features.index[-1].date()
        try:
            prior_history = price_store.read(ticker)
        except FileNotFoundError:
            prior_history = None
        if prior_history is not None and not prior_history.empty:
            prior_history = prior_history[prior_history.index.date < as_of]
            if prior_history.empty:
                prior_history = None
        try:
            row = build_inference_row(
                prior_day_features=prior_features.iloc[-1],
                today_open=quote["open"],
                yesterday_close=prev_close,
                snapshot_date=snapshot_date,
                as_of_date=as_of,
                prior_history=prior_history,
                spy_open=spy_open,
                spy_prev_close=spy_prev_close,
            )
        except StaleFeatureSnapshotError as exc:
            skipped.append({"ticker": ticker, "reason": str(exc)})
            continue

        scored_count += 1
        predicted_return = predict_signal(ensemble, row)
        if predicted_return > threshold:
            signals.append(
                BuySignal(
                    ticker=ticker,
                    predicted_return=predicted_return,
                    open_price=quote["open"],
                    snapshot_date=snapshot_date.isoformat(),
                )
            )

    signals.sort(key=lambda signal: signal.predicted_return, reverse=True)
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
