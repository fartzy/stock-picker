"""Thin wrapper around yfinance for batch OHLCV history pulls and live quotes.

History (ingestion, charts) stays on the chart/download API. This-morning
opens use the free cascade in `fetch_quotes`: Polygon snapshot if entitled,
Yahoo v7 quote snapshot (today's `regularMarketOpen`, not last), 1-minute
bars for names that have not printed yet, daily chart, then Finnhub /quote
for a small leftover set. Every source still requires the print's own date
to be today. A large overnight move is a real price, not a reason to drop
the ticker. Missing at 8:31 is "has not opened yet," not a hard failure.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from yfinance.const import _QUERY1_URL_
from yfinance.data import YfData

from stock_picker.log import get_logger
from stock_picker.parallel import BUCKET_SIZE, WORKERS, run_buckets

# Yahoo's v7 quote URL is a comma-separated list; 200 names is the same
# bucket size as morning scoring (~10 requests for a 2,000-name universe).
QUOTE_BATCH_SIZE = BUCKET_SIZE
QUOTE_WORKERS = WORKERS
# After a 200-name daily pull, retry names Yahoo omitted — first in 20s,
# then one ticker at a time so one thin name cannot kill a batch.
HISTORY_RETRY_BATCH = 20

logger = get_logger(__name__)
# 1-minute fallback is only for tickers the snapshot didn't date as today.
INTRADAY_FALLBACK_BATCH_SIZE = 100
REGULAR_SESSION_OPEN_MINUTES = 9 * 60 + 30
REGULAR_SESSION_CLOSE_MINUTES = 16 * 60


def _download_price_history_batch(
    tickers: list[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """One yfinance download. Empty / all-NaN names are omitted, not stored."""
    if not tickers:
        return {}
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )
    # yfinance always returns MultiIndex (ticker, field) columns when
    # group_by="ticker" is set, even for a single ticker.
    result = {}
    for ticker in tickers:
        if ticker not in raw:
            continue
        history = raw[ticker].dropna(how="all")
        if not history.empty:
            result[ticker] = history
    return result


def _retry_missing_histories(
    tickers: list[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Yahoo often drops thin names from a 2000-ticker pull. Try smaller groups."""
    recovered: dict[str, pd.DataFrame] = {}
    for start in range(0, len(tickers), HISTORY_RETRY_BATCH):
        chunk = tickers[start : start + HISTORY_RETRY_BATCH]
        recovered.update(_download_price_history_batch(chunk, period, interval))
    still = [ticker for ticker in tickers if ticker not in recovered]
    for ticker in still:
        recovered.update(_download_price_history_batch([ticker], period, interval))
    return recovered


def download_price_history(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV history for a batch of tickers.

    ~2,000 names go out in 200-ticker buckets (same size as morning quotes).
    Names Yahoo omitted are retried in groups of 20, then one at a time.
    Morning "no live quote" does not mean this nightly bar should be skipped.
    """
    if not tickers:
        return {}
    if len(tickers) > QUOTE_BATCH_SIZE:
        result: dict[str, pd.DataFrame] = {}
        for part in run_buckets(
            lambda chunk: _download_price_history_batch(list(chunk), period, interval),
            tickers,
            bucket_size=QUOTE_BATCH_SIZE,
            workers=QUOTE_WORKERS,
        ):
            result.update(part)
    else:
        result = _download_price_history_batch(tickers, period, interval)
    missing = [ticker for ticker in tickers if ticker not in result]
    if missing:
        logger.info("retrying %s tickers Yahoo dropped from the daily pull", len(missing))
        result.update(_retry_missing_histories(missing, period, interval))
        still = [ticker for ticker in tickers if ticker not in result]
        if still:
            logger.warning("still no daily history for %s: %s", len(still), ", ".join(still[:20]))
    return result


def _session_dates(index: pd.Index) -> pd.Index:
    """US-session calendar dates for a Yahoo daily-bar index.

    Daily bars can be tz-aware UTC midnight, which is the previous calendar
    date in US/Eastern -- convert first so "today" matches the cash session.
    """
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("America/New_York").tz_localize(None)
    return pd.Index(timestamps.date)


def quotes_from_history(histories: dict[str, pd.DataFrame], as_of: date) -> dict[str, dict]:
    """Pick today's open/last from a daily-bar pull. The bar's own date must
    be `as_of` -- taking iloc[-1] as "today" is how a stale pull silently
    reports yesterday's open as this morning's.
    """
    quotes = {}
    for ticker, history in histories.items():
        if history.empty or "Open" not in history.columns or "Close" not in history.columns:
            continue
        session_dates = _session_dates(history.index)
        today_mask = session_dates.to_numpy() == as_of
        if not today_mask.any():
            continue
        today_row = history.iloc[today_mask].iloc[-1]
        prior = history.iloc[~today_mask]
        if prior.empty:
            continue
        open_price = float(today_row["Open"])
        last_price = float(today_row["Close"])
        prev_close = float(prior["Close"].iloc[-1])
        if not (
            pd.notna(open_price)
            and pd.notna(last_price)
            and pd.notna(prev_close)
            and open_price > 0
            and prev_close > 0
        ):
            continue
        quotes[ticker] = {"open": open_price, "last": last_price, "prev_close": prev_close}
    return quotes


def _regular_session_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Rows whose clock time is in the US cash session (9:30-16:00 ET).

    Daily bars (midnight, no intraday clock) come back empty -- callers that
    want the whole day should not use this.
    """
    timestamps = pd.DatetimeIndex(history.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("America/New_York")
    else:
        timestamps = timestamps.tz_convert("America/New_York")
    minutes = timestamps.hour * 60 + timestamps.minute
    mask = (minutes >= REGULAR_SESSION_OPEN_MINUTES) & (minutes < REGULAR_SESSION_CLOSE_MINUTES)
    return history.iloc[mask]


def quotes_from_intraday(histories: dict[str, pd.DataFrame], as_of: date) -> dict[str, dict]:
    """Today's open = first regular-session 1-minute print on `as_of`.

    Previous close is the last print from a strictly earlier session date in
    the same pull (so this needs more than today's bars -- period="5d").
    """
    quotes = {}
    for ticker, history in histories.items():
        if history.empty or "Open" not in history.columns or "Close" not in history.columns:
            continue
        session_dates = _session_dates(history.index)
        today_mask = session_dates.to_numpy() == as_of
        if not today_mask.any():
            continue
        today = _regular_session_rows(history.iloc[today_mask])
        if today.empty:
            continue
        prior = history.iloc[~today_mask]
        if prior.empty:
            continue
        open_price = float(today["Open"].iloc[0])
        last_price = float(today["Close"].iloc[-1])
        prev_close = float(prior["Close"].iloc[-1])
        if not (
            pd.notna(open_price)
            and pd.notna(last_price)
            and pd.notna(prev_close)
            and open_price > 0
            and prev_close > 0
        ):
            continue
        quotes[ticker] = {"open": open_price, "last": last_price, "prev_close": prev_close}
    return quotes


def _session_date_from_unix(timestamp: object) -> date | None:
    """Yahoo quote timestamps are unix seconds in UTC; convert to the US
    cash-session calendar date (America/New_York)."""
    try:
        seconds = int(timestamp)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(ZoneInfo("America/New_York")).date()


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(number) or number <= 0:
        return None
    return number


def quotes_from_snapshot(raw_quotes: list[dict], as_of: date) -> dict[str, dict]:
    """Shape Yahoo v7 quote-snapshot rows into {ticker: open/last/prev_close}.

    Drops a ticker only when the snapshot isn't from `as_of`'s session
    (`regularMarketTime` is yesterday) or a required field is missing.
    A large overnight move is kept -- that's a real open. Pure, no network.
    """
    quotes = {}
    for raw in raw_quotes:
        ticker = raw.get("symbol")
        if not ticker:
            continue
        session_date = _session_date_from_unix(raw.get("regularMarketTime"))
        if session_date != as_of:
            continue
        open_price = _finite_positive(raw.get("regularMarketOpen"))
        last_price = _finite_positive(raw.get("regularMarketPrice"))
        prev_close = _finite_positive(raw.get("regularMarketPreviousClose"))
        if open_price is None or last_price is None or prev_close is None:
            continue
        quotes[ticker] = {"open": open_price, "last": last_price, "prev_close": prev_close}
    return quotes


def _snapshot_chunk(chunk: list[str]) -> list[dict]:
    """One Yahoo v7 batch. Own client per thread -- YfData is not shared."""
    params = {
        "symbols": ",".join(chunk),
        "formatted": "false",
        "lang": "en-US",
        "region": "US",
    }
    try:
        payload = YfData().get_raw_json(f"{_QUERY1_URL_}/v7/finance/quote", params=params)
    except Exception:
        return []
    return (payload or {}).get("quoteResponse", {}).get("result") or []


def fetch_quote_snapshots(tickers: list[str]) -> list[dict]:
    """Yahoo v7 quote snapshot for `tickers`, in 200-name buckets.

    One bad chunk (rate limit, crumb, delisted names) returns [] for that
    bucket and does not abort the rest.
    """
    snapshots: list[dict] = []
    for chunk_result in run_buckets(
        _snapshot_chunk,
        tickers,
        bucket_size=QUOTE_BATCH_SIZE,
        workers=QUOTE_WORKERS,
    ):
        snapshots.extend(chunk_result)
    return snapshots


def _missing(tickers: list[str], quotes: dict[str, dict]) -> list[str]:
    return [ticker for ticker in tickers if ticker not in quotes]


def copied_prior_open(open_price: float, prior_open: float) -> bool:
    """True when `open_price` is yesterday's Open to the cent.

    Yahoo's v7 snapshot often keeps `regularMarketOpen` on the prior session
    after `regularMarketTime` has flipped to today. That is Thursday's $23.18,
    not Friday's $26.27. A real gap vs previous *close* is kept.
    """
    return round(open_price, 2) == round(prior_open, 2)


def prior_session_opens(
    tickers: list[str],
    as_of: date,
    price_store=None,
) -> dict[str, float]:
    """Last PriceStore Open strictly before `as_of` -- yesterday's official Open."""
    from stock_picker.storage.price_store import PriceStore

    store = price_store if price_store is not None else PriceStore()
    opens: dict[str, float] = {}
    for ticker in tickers:
        try:
            history = store.read(ticker)
        except FileNotFoundError:
            continue
        if history.empty or "Open" not in history.columns:
            continue
        session_dates = _session_dates(history.index)
        prior = history.iloc[session_dates.to_numpy() < as_of]
        if prior.empty:
            continue
        open_price = float(prior["Open"].iloc[-1])
        if pd.notna(open_price) and open_price > 0:
            opens[ticker] = open_price
    return opens


def drop_copied_prior_opens(
    quotes: dict[str, dict],
    prior_opens: dict[str, float],
) -> dict[str, dict]:
    """Drop quotes whose open equals that ticker's prior-session Open."""
    kept = {}
    for ticker, quote in quotes.items():
        prior = prior_opens.get(ticker)
        if prior is not None and copied_prior_open(quote["open"], prior):
            logger.info(
                "drop copied prior open %s open=%s prior_open=%s",
                ticker,
                quote["open"],
                prior,
            )
            continue
        kept[ticker] = quote
    return kept


def _download_in_chunks(
    tickers: list[str], period: str, interval: str, chunk_size: int
) -> dict[str, pd.DataFrame]:
    def _one(chunk: list[str]) -> dict[str, pd.DataFrame]:
        return download_price_history(chunk, period=period, interval=interval)

    histories: dict[str, pd.DataFrame] = {}
    for chunk_result in run_buckets(_one, tickers, bucket_size=chunk_size, workers=QUOTE_WORKERS):
        histories.update(chunk_result)
    return histories


def fetch_yahoo_quotes(
    tickers: list[str],
    as_of: date,
    prior_opens: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Yahoo open = first 9:30 ET 1-minute print, not v7 regularMarketOpen.

    Snapshot Open at 8:31 CT is often yesterday's Open with today's
    timestamp (WRBY $23.18). 1-minute RTH is the cash print. Snapshot is
    leftover after that, then daily. Copied yesterday Open is always
    dropped -- YahooQuoteProvider does not pass prior_opens in.
    """
    if prior_opens is None:
        prior_opens = prior_session_opens(tickers, as_of)
    quotes = quotes_from_intraday(
        _download_in_chunks(tickers, period="5d", interval="1m", chunk_size=INTRADAY_FALLBACK_BATCH_SIZE),
        as_of=as_of,
    )
    missing = _missing(tickers, quotes)
    if missing:
        leftover = quotes_from_snapshot(fetch_quote_snapshots(missing), as_of=as_of)
        leftover = drop_copied_prior_opens(leftover, prior_opens)
        quotes.update(leftover)
        missing = _missing(tickers, quotes)
    if missing:
        quotes.update(
            quotes_from_history(
                _download_in_chunks(missing, period="5d", interval="1d", chunk_size=QUOTE_BATCH_SIZE),
                as_of=as_of,
            )
        )
    return drop_copied_prior_opens(quotes, prior_opens)


def fetch_quotes(
    tickers: list[str],
    as_of: date | None = None,
    prior_opens: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Facade -- Polygon then Yahoo then Finnhub. See quote_providers.fetch_quotes."""
    from stock_picker.ingestion.quote_providers import fetch_quotes as fetch_session_quotes

    return fetch_session_quotes(tickers, as_of=as_of)
