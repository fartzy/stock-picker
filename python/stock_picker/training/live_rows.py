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

from stock_picker.parallel import BUCKET_SIZE, WORKERS, run_buckets
from stock_picker.features.structure import fill_cluster_overnight_gaps
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.ticker_blacklist_store import blacklisted_tickers
from stock_picker.training.inference import StaleFeatureSnapshotError, build_inference_row


@dataclass
class LiveRow:
    ticker: str
    skipped: dict | None = None
    open_price: float | None = None
    prev_close: float | None = None
    snapshot_date: str | None = None
    row: pd.DataFrame | None = None


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

    Thread buckets of 100 -- parquet I/O, not GIL-bound LightGBM.
    Rank and Fit both consume this list.
    """
    blocked = blacklisted_tickers()
    earning_set = set(earnings)

    def _chunk(chunk: list[str]) -> list[LiveRow]:
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
                blocked=blocked,
            )
            for ticker in chunk
        ]

    rows: list[LiveRow] = []
    for bucket in run_buckets(_chunk, tickers, bucket_size=bucket_size, workers=workers):
        rows.extend(bucket)
    fill_cluster_overnight_gaps(rows, quotes)
    return rows
