"""One open-known inference matrix for Rank and Fit.

Rank and Fit used to each rebuild ~2,000 parquet + seasonality rows.
That pandas work is the morning wall -- LightGBM predict is cheap after it.
Build the rows once, in process buckets, then both models only predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_picker.parallel import BUCKET_SIZE, WORKERS, run_buckets_processes, split_buckets
from stock_picker.features.structure import fill_cluster_overnight_gaps
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.ticker_blacklist_store import blacklisted_tickers
from stock_picker.training.inference import StaleFeatureSnapshotError, build_inference_row
from stock_picker.features.kronos import CONTRACT_ATTR, KRONOS_COLUMNS, live_inputs, target_has_known_split
from stock_picker.storage.kronos_store import KronosStore


@dataclass
class LiveRow:
    ticker: str
    skipped: dict | None = None
    open_price: float | None = None
    prev_close: float | None = None
    snapshot_date: str | None = None
    row: pd.DataFrame | None = None
    as_of: str | None = None
    prior_history: pd.DataFrame | None = None
    historical_replay: bool = False
    known_target_split: bool = False


def _index_dates(index: pd.Index) -> np.ndarray:
    """Calendar dates without a Python date object per row."""
    values = pd.DatetimeIndex(index)
    if values.tz is not None:
        values = values.tz_convert("America/New_York").tz_localize(None)
    return values.normalize().date


def prepare_one(
    ticker: str,
    quotes: dict[str, dict],
    earnings: set[str],
    feature_store: FeatureStore,
    price_store: PriceStore,
    as_of: date,
    spy_open: float | None,
    spy_prev_close: float | None,
    blocked: set[str] | None = None,
    excluded_features: frozenset[str] | None = None,
) -> LiveRow:
    if ticker in (blocked if blocked is not None else blacklisted_tickers()):
        return LiveRow(
            ticker,
            skipped={"ticker": ticker, "reason": "blacklisted"},
        )
    if ticker in earnings:
        return LiveRow(
            ticker,
            skipped={
                "ticker": ticker,
                "reason": "earnings on or since the prior session -- news day, not a pattern day",
            },
        )
    quote = quotes.get(ticker)
    if quote is None:
        return LiveRow(ticker, skipped={"ticker": ticker, "reason": "no live quote available"})
    prev_close = quote.get("prev_close")
    if prev_close is None:
        return LiveRow(ticker, skipped={"ticker": ticker, "reason": "no previous close available"})
    try:
        prior_features = feature_store.read(ticker)
    except FileNotFoundError:
        return LiveRow(ticker, skipped={"ticker": ticker, "reason": "no feature history"})
    feature_dates = _index_dates(prior_features.index)
    prior_features = prior_features.iloc[feature_dates < as_of]
    if prior_features.empty:
        return LiveRow(ticker, skipped={"ticker": ticker, "reason": "no feature snapshot before this morning"})
    snapshot_date = feature_dates[feature_dates < as_of][-1]
    try:
        prior_history = price_store.read(ticker)
    except FileNotFoundError:
        prior_history = None
    known_target_split = target_has_known_split(prior_history, as_of.isoformat())
    if prior_history is not None and not prior_history.empty:
        prior_history = prior_history.iloc[_index_dates(prior_history.index) < as_of]
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
            excluded_features=excluded_features,
        )
    except StaleFeatureSnapshotError as extra:
        return LiveRow(ticker, skipped={"ticker": ticker, "reason": str(extra)})
    return LiveRow(
        ticker,
        open_price=quote["open"],
        prev_close=float(prev_close) if prev_close is not None else None,
        snapshot_date=snapshot_date.isoformat(),
        row=row,
        as_of=as_of.isoformat(),
        prior_history=prior_history.tail(128) if prior_history is not None else None,
        known_target_split=known_target_split,
    )


def model_frame(rows: list[LiveRow], ensemble, store: KronosStore | None = None) -> pd.DataFrame:
    """Share expensive base rows, materialize cheap K inputs for each saved head.

    Its paired SVM may consume K even when the final head does not.
    """
    from stock_picker.training.ensemble import ensemble_kronos_contract

    frame = pd.concat([item.row for item in rows], ignore_index=True)
    contract = ensemble_kronos_contract(ensemble)
    if contract is None:
        return frame
    cache = store if store is not None else KronosStore()
    inputs = [live_inputs(item.ticker, item.as_of, item.open_price, item.prev_close,
                          item.prior_history, cache, contract,
                          allow_reconstructed=item.historical_replay,
                          known_target_split=item.known_target_split) for item in rows]
    for column in KRONOS_COLUMNS:
        frame[column] = [values[column] for values in inputs]
    frame.attrs[CONTRACT_ATTR] = contract
    frame.attrs["kronos_usage"] = "counterfactual_replay" if any(item.historical_replay for item in rows) else "pre_open_observation"
    return frame


def with_live_model_context(item: LiveRow) -> pd.DataFrame:
    """Carry one scenario's read-only price context to its pinned day-model heads.

    This ephemeral metadata never enters historical feature snapshots. Use
    plain containers so pandas can copy/compare attrs without DataFrame equality.
    """
    frame = item.row.copy()
    history = item.prior_history
    frame.attrs["kronos_live_context"] = {
        "ticker": item.ticker, "as_of": item.as_of, "open_price": item.open_price,
        "prev_close": item.prev_close,
        "known_target_split": item.known_target_split,
        "sessions": [day.isoformat() for day in _index_dates(history.index)] if history is not None else [],
        "bars": history[["Open", "High", "Low", "Close", "Volume"]].to_dict("list") if history is not None else {},
    }
    return frame


def frame_for_saved_model(frame: pd.DataFrame, ensemble) -> pd.DataFrame:
    """Historical rows keep their pinned attrs; live scenarios resolve each head."""
    context = frame.attrs.get("kronos_live_context")
    if context is None:
        return frame
    if len(frame) != 1:
        raise ValueError("live model context must describe exactly one scenario row")
    history = pd.DataFrame(context["bars"], index=pd.DatetimeIndex(context["sessions"]))
    item = LiveRow(context["ticker"], row=frame, as_of=context["as_of"],
                   open_price=context["open_price"], prev_close=context["prev_close"], prior_history=history,
                   known_target_split=context.get("known_target_split", False))
    result = model_frame([item], ensemble)
    result.index = frame.index
    return result


@dataclass
class _BucketPayload:
    """One process-pool call's worth of work: a ticker chunk plus every
    read-only value `prepare_one` needs. `run_buckets_processes` requires a
    picklable, module-level `work` function -- a closure capturing these
    (as `prepare_live_rows` used to, back when this ran on threads) can't
    cross a process boundary, so they travel as plain dataclass fields
    instead. `feature_store`/`price_store` are safe to pickle as-is: each
    is just a wrapped `Path` (see storage/feature_store.py,
    storage/price_store.py), not a live connection or file handle, so a
    worker's copy resolves the exact same on-disk location -- including a
    test's `data_dir` override.
    """

    tickers: list[str]
    quotes: dict[str, dict]
    earnings: set[str]
    feature_store: FeatureStore
    price_store: PriceStore
    as_of: date
    spy_open: float | None
    spy_prev_close: float | None
    blocked: set[str]
    excluded_features: frozenset[str] | None


def _prepare_bucket(payload: _BucketPayload) -> list[LiveRow]:
    return [
        prepare_one(
            ticker,
            quotes=payload.quotes,
            earnings=payload.earnings,
            feature_store=payload.feature_store,
            price_store=payload.price_store,
            as_of=payload.as_of,
            spy_open=payload.spy_open,
            spy_prev_close=payload.spy_prev_close,
            blocked=payload.blocked,
            excluded_features=payload.excluded_features,
        )
        for ticker in payload.tickers
    ]


def prepare_live_rows(
    tickers: list[str],
    quotes: dict[str, dict],
    earnings: set[str],
    feature_store: FeatureStore,
    price_store: PriceStore,
    as_of: date,
    spy_open: float | None,
    spy_prev_close: float | None,
    bucket_size: int = BUCKET_SIZE,
    workers: int = WORKERS,
    excluded_features: frozenset[str] | None = None,
) -> list[LiveRow]:
    """Rebuild open-known rows once for the universe.

    Process buckets of 100: per-ticker cost here is CPU-bound pandas
    (groupby+expanding+transform in features/open_pattern_seasonality.py),
    not the parquet I/O this was originally sized for -- Python threads
    don't parallelize CPU work (the GIL blocks it), so this used to run
    close to sequential despite the bucket machinery. Measured: a 50-name
    sample through the old thread path (8.76s) was not faster than plain
    sequential (8.33s) -- see training/profile_live_rows.py.
    Rank and Fit both consume this list.

    `excluded_features` (currently-pruned columns) skips their open-known
    recomputation per ticker -- every trainer's own `feature_columns()`
    drops them before prediction anyway, so computing them here is pure
    waste on the one path with a real morning-window time budget. Defaults
    to None (compute everything) rather than reading `pruned_features()`
    itself, so callers control freshness explicitly instead of this
    function reading global state on every invocation.
    """
    blocked = blacklisted_tickers()
    earning_set = set(earnings)
    payloads = [
        _BucketPayload(
            tickers=chunk,
            quotes=quotes,
            earnings=earning_set,
            feature_store=feature_store,
            price_store=price_store,
            as_of=as_of,
            spy_open=spy_open,
            spy_prev_close=spy_prev_close,
            blocked=blocked,
            excluded_features=excluded_features,
        )
        for chunk in split_buckets(tickers, bucket_size)
    ]

    rows: list[LiveRow] = []
    for bucket in run_buckets_processes(_prepare_bucket, payloads, workers=workers):
        rows.extend(bucket)
    fill_cluster_overnight_gaps(rows, quotes)
    return rows
