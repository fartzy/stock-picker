"""One open-known inference matrix for Rank and Fit.

Rank and Fit used to each rebuild ~2,000 parquet + seasonality rows.
That pandas work is the morning wall -- LightGBM predict is cheap after it.
Build the rows once, in process buckets, then both models only predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from stock_picker.parallel import BUCKET_SIZE, WORKERS, run_buckets_processes, split_buckets
from stock_picker.features.structure import fill_cluster_overnight_gaps
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.training.inference import StaleFeatureSnapshotError, build_inference_row


@dataclass
class LiveRow:
    ticker: str
    skipped: dict | None = None
    open_price: float | None = None
    prev_close: float | None = None
    snapshot_date: str | None = None
    row: pd.DataFrame | None = None


def prepare_one(
    ticker: str,
    quotes: dict[str, dict],
    earnings: set[str],
    feature_store: FeatureStore,
    price_store: PriceStore,
    as_of: date,
    spy_open: float | None,
    spy_prev_close: float | None,
) -> LiveRow:
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
        return LiveRow(ticker, skipped={"ticker": ticker, "reason": str(extra)})
    return LiveRow(
        ticker,
        open_price=quote["open"],
        prev_close=float(prev_close) if prev_close is not None else None,
        snapshot_date=snapshot_date.isoformat(),
        row=row,
    )


def prepare_bucket_payload(payload: tuple) -> list[LiveRow]:
    """Module-level so ProcessPoolExecutor can pickle it under spawn."""
    (
        tickers,
        quotes,
        earnings,
        feature_dir,
        price_dir,
        as_of_iso,
        spy_open,
        spy_prev_close,
    ) = payload
    as_of = date.fromisoformat(as_of_iso)
    feature_store = FeatureStore(data_dir=feature_dir)
    price_store = PriceStore(data_dir=price_dir)
    earning_set = set(earnings)
    return [
        prepare_one(
            ticker,
            quotes=quotes,
            earnings=earning_set,
            feature_store=feature_store,
            price_store=price_store,
            as_of=as_of,
            spy_open=spy_open,
            spy_prev_close=spy_prev_close,
        )
        for ticker in tickers
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
) -> list[LiveRow]:
    """Rebuild open-known rows once for the universe.

    10 process buckets of 200. Rank and Fit both consume this list.
    """
    chunks = split_buckets(tickers, bucket_size)
    payloads = [
        (
            chunk,
            quotes,
            list(earnings),
            str(feature_store._data_dir),
            str(price_store._data_dir),
            as_of.isoformat(),
            spy_open,
            spy_prev_close,
        )
        for chunk in chunks
    ]
    rows: list[LiveRow] = []
    for bucket in run_buckets_processes(prepare_bucket_payload, payloads, workers=workers):
        rows.extend(bucket)
    fill_cluster_overnight_gaps(rows, quotes)
    return rows
