"""Live "what should I buy this morning" scoring -- the app's whole premise in
one function: score every active ticker's current-morning quote through the
persisted ensemble and report which ones clear a confidence threshold, and why.

Every building block here already exists and is independently tested --
`inference.py` builds the lookahead-safe row and guards against stale/implausible
data, `quotes.py` batch-fetches live quotes in a single call, `importance.py`
already computes the model's blended feature importance. This module is the per-ticker loop that ties them together, skipping (and
recording why) rather than crashing on any one ticker's bad data. Scoring
runs in 200-name buckets via `stock_picker.parallel` -- same size as the
Yahoo quote batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import partial
from typing import Callable

import pandas as pd

from stock_picker.features.quotes import fetch_ticker_quotes
from stock_picker.parallel import BUCKET_SIZE, WORKERS, run_buckets
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble, predict_ensemble
from stock_picker.training.importance import ensemble_importance
from stock_picker.training.inference import (
    StaleFeatureSnapshotError,
    build_inference_row,
)
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
SCORE_BUCKET = BUCKET_SIZE
SCORE_WORKERS = WORKERS


@dataclass
class BuySignal:
    ticker: str
    predicted_return: float
    open_price: float
    snapshot_date: str
    news_flag: str | None = None


@dataclass
class BuySignalResult:
    as_of: str
    threshold: float
    signals: list[BuySignal] = field(default_factory=list)
    scored_count: int = 0
    skipped: list[dict] = field(default_factory=list)
    top_drivers: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class TickerScore:
    """One name after parquet + open-known row + predict.

    skipped is set XOR scored. A scored name that misses the threshold
    still counts in scored_count; signal is None until it clears.
    """

    ticker: str
    skipped: dict | None = None
    scored: bool = False
    signal: BuySignal | None = None


def _prepare_one(
    ticker: str,
    quotes: dict[str, dict],
    earnings: set[str],
    feature_store: FeatureStore,
    price_store: PriceStore,
    as_of: date,
    spy_open: float | None,
    spy_prev_close: float | None,
) -> tuple[TickerScore | None, dict | None]:
    """Skip, or the open-known row ready for a bucket-wide predict.

    Returns (skip, None) or (None, {ticker, open, snapshot_date, row}).
    Predict is not here -- one LightGBM call per 200-name bucket.
    """
    if ticker in earnings:
        return (
            TickerScore(
                ticker,
                skipped={
                    "ticker": ticker,
                    "reason": "earnings on or since the prior session -- news day, not a pattern day",
                },
            ),
            None,
        )
    quote = quotes.get(ticker)
    if quote is None:
        return TickerScore(ticker, skipped={"ticker": ticker, "reason": "no live quote available"}), None
    prev_close = quote.get("prev_close")
    if prev_close is None:
        return TickerScore(ticker, skipped={"ticker": ticker, "reason": "no previous close available"}), None
    try:
        prior_features = feature_store.read(ticker)
    except FileNotFoundError:
        return TickerScore(ticker, skipped={"ticker": ticker, "reason": "no feature history"}), None
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
    except StaleFeatureSnapshotError as extra:
        return TickerScore(ticker, skipped={"ticker": ticker, "reason": str(extra)}), None
    return None, {
        "ticker": ticker,
        "open": quote["open"],
        "snapshot_date": snapshot_date.isoformat(),
        "row": row,
    }


def _score_bucket(
    tickers,
    quotes,
    earnings,
    ensemble: Ensemble,
    feature_store,
    price_store,
    as_of,
    threshold: float,
    spy_open,
    spy_prev_close,
) -> list[TickerScore]:
    prepared: list[dict] = []
    results: list[TickerScore] = []
    for ticker in tickers:
        skip, ready = _prepare_one(
            ticker,
            quotes=quotes,
            earnings=earnings,
            feature_store=feature_store,
            price_store=price_store,
            as_of=as_of,
            spy_open=spy_open,
            spy_prev_close=spy_prev_close,
        )
        if skip is not None:
            results.append(skip)
            continue
        prepared.append(ready)
    if not prepared:
        return results
    frame = pd.concat([item["row"] for item in prepared], ignore_index=True)
    predicted = predict_ensemble(ensemble, frame)
    for item, predicted_return in zip(prepared, predicted):
        predicted_return = float(predicted_return)
        if predicted_return > threshold:
            results.append(
                TickerScore(
                    item["ticker"],
                    scored=True,
                    signal=BuySignal(
                        ticker=item["ticker"],
                        predicted_return=predicted_return,
                        open_price=item["open"],
                        snapshot_date=item["snapshot_date"],
                    ),
                )
            )
        else:
            results.append(TickerScore(item["ticker"], scored=True))
    return results


def _score_universe(tickers: list[str], **kwargs) -> tuple[list[BuySignal], list[dict], int]:
    signals: list[BuySignal] = []
    skipped: list[dict] = []
    scored_count = 0
    for bucket in run_buckets(
        partial(_score_bucket, **kwargs),
        tickers,
        bucket_size=SCORE_BUCKET,
        workers=SCORE_WORKERS,
    ):
        for result in bucket:
            if result.skipped is not None:
                skipped.append(result.skipped)
                continue
            if result.scored:
                scored_count += 1
            if result.signal is not None:
                signals.append(result.signal)
    return signals, skipped, scored_count


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

    signals, skipped, scored_count = _score_universe(
        tickers=tickers,
        quotes=quotes,
        earnings=earnings,
        ensemble=ensemble,
        feature_store=feature_store,
        price_store=price_store,
        as_of=as_of,
        threshold=threshold,
        spy_open=spy_open,
        spy_prev_close=spy_prev_close,
    )

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
