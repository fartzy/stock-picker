"""Refresh OHLCV for the already-active universe -- no recap / market-cap rebuild."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from stock_picker.ingestion.finnhub_client import (
    MIN_SECONDS_BETWEEN_CALLS,
    fetch_finnhub_candles,
)
from stock_picker.ingestion.session import completed_sessions, last_completed_session_date
from stock_picker.ingestion.yfinance_client import download_price_history
from stock_picker.log import get_logger
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore

logger = get_logger(__name__)

FINNHUB_BACKFILL_LOOKBACK_DAYS = 365


def _index_date(value) -> date | None:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York").tz_localize(None)
    return timestamp.date()


def last_session_date(history: pd.DataFrame) -> date | None:
    if history is None or history.empty:
        return None
    return _index_date(history.index[-1])


def tickers_missing_session(tickers: list[str], store: PriceStore, cutoff: date) -> list[str]:
    """Active names whose stored daily bars do not include `cutoff`."""
    missing = []
    for ticker in tickers:
        try:
            history = store.read(ticker)
        except FileNotFoundError:
            missing.append(ticker)
            continue
        last = last_session_date(history)
        if last is None or last < cutoff:
            missing.append(ticker)
    return missing


def history_from_finnhub_bars(bars: list[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame(bars)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame["Adj Close"] = frame["Close"]
    return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def merge_price_history(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return existing if existing is not None else pd.DataFrame()
    if existing is None or existing.empty:
        return incoming
    combined = pd.concat([existing, incoming])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def backfill_missing_sessions(
    tickers: list[str],
    store: PriceStore,
    cutoff: date,
    fetch_candles=fetch_finnhub_candles,
    sleep_seconds: float = MIN_SECONDS_BETWEEN_CALLS,
) -> dict[str, int]:
    """Finnhub daily candles for names Yahoo still omitted after the 1y pull.

    One leftover name is one call. A 429/`None` fetch is skipped so a later
    nightly can retry. Written bars are dated on or before `cutoff`.
    """
    missing = tickers_missing_session(tickers, store, cutoff)
    written = 0
    if not missing:
        logger.info("evening price backfill -- every ticker has %s", cutoff)
        return {"missing": 0, "written": 0}
    logger.info("evening price backfill %s tickers still missing %s", len(missing), cutoff)
    for index, ticker in enumerate(missing):
        if index and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            existing = store.read(ticker)
        except FileNotFoundError:
            existing = None
        last = last_session_date(existing) if existing is not None else None
        start = (last + timedelta(days=1)) if last is not None else cutoff - timedelta(days=FINNHUB_BACKFILL_LOOKBACK_DAYS)
        try:
            bars = fetch_candles(ticker, start, cutoff)
        except Exception:
            logger.exception("finnhub candle backfill failed ticker=%s", ticker)
            continue
        if bars is None:
            logger.warning("finnhub candle backfill failed ticker=%s", ticker)
            continue
        incoming = history_from_finnhub_bars(bars)
        if incoming.empty:
            continue
        timestamps = pd.DatetimeIndex(incoming.index)
        if timestamps.tz is not None:
            timestamps = timestamps.tz_convert("America/New_York").tz_localize(None)
        incoming = incoming.iloc[pd.Index(timestamps.date) <= cutoff]
        if incoming.empty:
            continue
        merged = merge_price_history(existing, incoming)
        if merged.empty:
            continue
        store.write(ticker, merged)
        written += 1
    still = tickers_missing_session(tickers, store, cutoff)
    logger.info("evening price backfill wrote %s, still missing %s", written, len(still))
    return {"missing": len(missing), "written": written, "still_missing": len(still)}


def refresh_prices() -> None:
    tickers = UniverseStore().active_tickers()
    logger.info("refreshing prices for %s tickers", len(tickers))
    history = download_price_history(tickers, period="1y", interval="1d")
    logger.info("downloaded %s", len(history))
    store = PriceStore()
    cutoff = last_completed_session_date()
    written = 0
    for ticker, frame in history.items():
        finished = completed_sessions(frame)
        if finished.empty:
            continue
        store.write(ticker, finished)
        written += 1
    logger.info("wrote %s through %s", written, cutoff)
    backfill_missing_sessions(tickers, store, cutoff)
    sample = store.read("AAPL")
    logger.info("AAPL last %s n=%s", sample.index[-1].date(), len(sample))


def main() -> None:
    refresh_prices()


if __name__ == "__main__":
    main()
