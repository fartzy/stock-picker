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

# Yahoo's v7 quote endpoint accepts a comma-separated symbols list; keep
# chunks well under typical URL-length limits and retry-friendly.
QUOTE_BATCH_SIZE = 200
# 1-minute fallback is only for tickers the snapshot didn't date as today.
INTRADAY_FALLBACK_BATCH_SIZE = 100
REGULAR_SESSION_OPEN_MINUTES = 9 * 60 + 30
REGULAR_SESSION_CLOSE_MINUTES = 16 * 60


def download_price_history(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV history for a batch of tickers.

    Returns a mapping of ticker -> DataFrame with columns
    [Open, High, Low, Close, Adj Close, Volume], indexed by date.
    """
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    # yfinance always returns MultiIndex (ticker, field) columns when group_by="ticker"
    # is set, even for a single ticker -- always slice per ticker, no single-ticker
    # special case. A ticker yfinance couldn't fetch (delisted, transient API failure,
    # etc.) comes back as an all-NaN slice -- exclude it rather than storing an empty
    # DataFrame that would break downstream assumptions of at least one row.
    result = {}
    for ticker in tickers:
        if ticker not in raw:
            continue
        history = raw[ticker].dropna(how="all")
        if not history.empty:
            result[ticker] = history
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


def fetch_quote_snapshots(tickers: list[str]) -> list[dict]:
    """Yahoo v7 quote snapshot for `tickers`, chunked. Empty list on a
    chunk that Yahoo doesn't return rather than aborting the whole scan."""
    if not tickers:
        return []
    client = YfData()
    snapshots: list[dict] = []
    for start in range(0, len(tickers), QUOTE_BATCH_SIZE):
        chunk = tickers[start : start + QUOTE_BATCH_SIZE]
        params = {
            "symbols": ",".join(chunk),
            "formatted": "false",
            "lang": "en-US",
            "region": "US",
        }
        try:
            payload = client.get_raw_json(f"{_QUERY1_URL_}/v7/finance/quote", params=params)
        except Exception:
            # Same contract as download_price_history: one bad chunk (rate
            # limit, crumb, delisted names) must not abort the rest.
            continue
        result = (payload or {}).get("quoteResponse", {}).get("result") or []
        snapshots.extend(result)
    return snapshots


def _missing(tickers: list[str], quotes: dict[str, dict]) -> list[str]:
    return [ticker for ticker in tickers if ticker not in quotes]


def _download_in_chunks(
    tickers: list[str], period: str, interval: str, chunk_size: int
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for start in range(0, len(tickers), chunk_size):
        histories.update(
            download_price_history(tickers[start : start + chunk_size], period=period, interval=interval)
        )
    return histories


def fetch_yahoo_quotes(tickers: list[str], as_of: date) -> dict[str, dict]:
    """Yahoo-only cascade: quote snapshot, then 1-minute RTH bars, then daily.

    Snapshot is today's official open (`regularMarketOpen`) as soon as Yahoo
    has it -- usually seconds after the print, in 200-ticker batches, so
    ~2000 names is about 10 requests. 1-minute bars are only for names the
    snapshot didn't date as today (has not opened / delayed daily candle).
    """
    quotes = quotes_from_snapshot(fetch_quote_snapshots(tickers), as_of=as_of)
    missing = _missing(tickers, quotes)
    if missing:
        quotes.update(
            quotes_from_intraday(
                _download_in_chunks(missing, period="5d", interval="1m", chunk_size=INTRADAY_FALLBACK_BATCH_SIZE),
                as_of=as_of,
            )
        )
        missing = _missing(tickers, quotes)
    if missing:
        quotes.update(
            quotes_from_history(
                _download_in_chunks(missing, period="5d", interval="1d", chunk_size=QUOTE_BATCH_SIZE),
                as_of=as_of,
            )
        )
    return quotes


def fetch_quotes(tickers: list[str], as_of: date | None = None) -> dict[str, dict]:
    """Today's regular-session open + latest price + previous close.

    Free path, in order, each requiring the print to be dated `as_of`:

    1. Polygon full-market snapshot -- one call, only if the key is entitled
       (Starter/free is 403; then this is a no-op).
    2. Yahoo quote snapshot -- bulk, today's `regularMarketOpen`.
    3. Yahoo 1-minute bars -- first 9:30 ET print, for names still missing.
    4. Yahoo daily chart -- last resort if a today bar exists.
    5. Finnhub /quote -- leftover only (rate-limited, not 2000 names).

    Names still missing after that have not printed an open yet -- the
    caller can retry later; we do not invent an open from last trade.
    """
    from stock_picker.ingestion.finnhub_client import fetch_finnhub_quotes
    from stock_picker.ingestion.polygon_client import fetch_polygon_quotes

    as_of = as_of or date.today()
    quotes = fetch_polygon_quotes(tickers, as_of=as_of)
    missing = _missing(tickers, quotes)
    if missing:
        quotes.update(fetch_yahoo_quotes(missing, as_of=as_of))
        missing = _missing(tickers, quotes)
    if missing:
        quotes.update(fetch_finnhub_quotes(missing, as_of=as_of))
    return quotes
