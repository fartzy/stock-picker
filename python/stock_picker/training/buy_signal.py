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
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.importance import ensemble_importance
from stock_picker.training.inference import (
    ImplausibleGapError,
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
    quote_fetcher: Callable[[list[str]], dict[str, dict]] = fetch_ticker_quotes,
) -> BuySignalResult:
    as_of = as_of or date.today()
    universe_store = universe_store or UniverseStore()
    feature_store = feature_store or FeatureStore()
    model_store = model_store or ModelStore()

    if not model_store.exists(MODEL_NAME):
        return BuySignalResult(
            as_of=as_of.isoformat(),
            threshold=threshold,
            skipped=[{"ticker": NO_MODEL_SENTINEL, "reason": "no trained model persisted yet"}],
        )

    ensemble = model_store.read(MODEL_NAME)
    tickers = universe_store.active_tickers()
    quotes = quote_fetcher(tickers)

    signals: list[BuySignal] = []
    skipped: list[dict] = []
    scored_count = 0
    for ticker in tickers:
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
            row = build_inference_row(
                prior_day_features=prior_features.iloc[-1],
                today_open=quote["open"],
                yesterday_close=prev_close,
                snapshot_date=snapshot_date,
                as_of_date=as_of,
            )
        except (StaleFeatureSnapshotError, ImplausibleGapError) as exc:
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
