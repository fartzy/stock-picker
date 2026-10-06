"""Polygon.io full-market snapshot for this-morning opens.

One HTTP call returns every US ticker's current session state -- the right
shape for scoring ~2000 names at the open, instead of 2000 (or even 10)
per-ticker requests. History ingestion stays on yfinance; this is only the
live open/last/prev-close path.

Key lookup, first hit wins: `POLYGON_API_KEY`, repo `.env`, then
`~/.config/api/polygon-massive.txt` (`default_key=`). Without a key,
fetch_quotes falls through to Yahoo. A ticker is kept only when the
snapshot is dated `as_of` (updated nanoseconds, else last-trade timestamp)
and has a positive open -- a large overnight move is a real print, not a
reason to drop it. Illiquid names with no day.open yet are skipped until
`day.o` prints.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

POLYGON_SNAPSHOT_URL = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"
POLYGON_KEY_FILE = Path.home() / ".config" / "api" / "polygon-massive.txt"
SNAPSHOT_TIMEOUT_SECONDS = 30


def _repo_root() -> Path:
    """Directory the user ran from (`bazel run` sets BUILD_WORKING_DIRECTORY
    because the process cwd is a sandbox with no `.env`)."""
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd()))


def _assigned_value(line: str, name: str) -> str | None:
    prefix = f"{name}="
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    value = stripped[len(prefix) :].strip().strip("'\"")
    return value or None


def _key_from_dotenv(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        value = _assigned_value(line, POLYGON_API_KEY_ENV)
        if value:
            return value
    return None


def _key_from_config_file(path: Path) -> str | None:
    """`default_key=...` from ~/.config/api/polygon-massive.txt, or a file
    whose only non-comment line is the key itself."""
    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    for line in lines:
        value = _assigned_value(line, "default_key") or _assigned_value(line, POLYGON_API_KEY_ENV)
        if value:
            return value
    nonempty = [line.strip().strip("'\"") for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(nonempty) == 1 and "=" not in nonempty[0]:
        return nonempty[0]
    return None


def polygon_api_key(key_file: Path | None = None) -> str | None:
    key = os.environ.get(POLYGON_API_KEY_ENV, "").strip()
    if key:
        return key
    from_dotenv = _key_from_dotenv(_repo_root() / ".env")
    if from_dotenv:
        return from_dotenv
    return _key_from_config_file(key_file or POLYGON_KEY_FILE)


def _eastern_datetime_from_ns(timestamp_ns: object) -> datetime | None:
    """Convert Polygon's Unix nanoseconds to a readable Eastern timestamp."""
    try:
        nanos = int(timestamp_ns)
        if nanos <= 0:
            return None
        return datetime.fromtimestamp(nanos / 1_000_000_000, tz=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _session_date_from_ns(timestamp_ns: object) -> date | None:
    """Polygon timestamps are Unix nanoseconds; validate in the US session timezone."""
    timestamp = _eastern_datetime_from_ns(timestamp_ns)
    return timestamp.date() if timestamp is not None else None


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:  # NaN or non-positive
        return None
    return number


def _item_session_date(item: dict) -> date | None:
    updated = item.get("updated")
    if updated:
        parsed = _session_date_from_ns(updated)
        if parsed is not None:
            return parsed
    last_trade = item.get("lastTrade") or {}
    return _session_date_from_ns(last_trade.get("t"))


def _matched_ticker(item: dict, wanted: set[str]) -> str | None:
    polygon_ticker = item.get("ticker")
    if not isinstance(polygon_ticker, str):
        return None
    # Polygon spells class shares BRK.B; our universe uses BRK-B.
    ticker = polygon_ticker if polygon_ticker in wanted else polygon_ticker.replace(".", "-")
    return ticker if ticker in wanted else None


def _quote_and_rejection(item: dict, as_of: date) -> tuple[dict | None, str | None]:
    if _item_session_date(item) != as_of:
        return None, "not_dated_today"
    day = item.get("day") or {}
    prev_day = item.get("prevDay") or {}
    last_trade = item.get("lastTrade") or {}
    open_price = _finite_positive(day.get("o"))
    if open_price is None:
        return None, "no_day_open"
    prev_close = _finite_positive(prev_day.get("c"))
    if prev_close is None:
        return None, "no_prev_close"
    # The opening trade itself is the latest known price if neither the
    # separate last-trade field nor the daily close has populated yet.
    last_price = _finite_positive(last_trade.get("p")) or _finite_positive(day.get("c")) or open_price
    return {"open": open_price, "last": last_price, "prev_close": prev_close}, None


def quotes_from_polygon_snapshot(
    raw_tickers: list[dict],
    wanted: set[str],
    as_of: date,
) -> dict[str, dict]:
    """Shape Polygon snapshot rows into {ticker: open/last/prev_close}.

    `day.o` is today's official open once the opening cross has printed.
    `lastTrade.p` fills `last` when available, then `day.c`, then the known
    opening trade itself. If `day.o` is still empty, we do not invent an
    open from lastTrade -- that's a last print, not the open. Previous
    close is `prevDay.c`. Pure -- no network.
    """
    quotes = {}
    for item in raw_tickers:
        ticker = _matched_ticker(item, wanted)
        if ticker is None:
            continue
        quote, _ = _quote_and_rejection(item, as_of)
        if quote is not None:
            quotes[ticker] = quote
    return quotes


def closing_quotes_from_polygon_snapshot(
    raw_tickers: list[dict], wanted: set[str], as_of: date
) -> dict[str, dict]:
    """Use today's aggregate close, never a later after-hours last trade."""
    quotes = {}
    for item in raw_tickers:
        ticker = _matched_ticker(item, wanted)
        if ticker is None or _item_session_date(item) != as_of:
            continue
        day = item.get("day") or {}
        open_price = _finite_positive(day.get("o"))
        close_price = _finite_positive(day.get("c"))
        if open_price is not None and close_price is not None:
            quotes[ticker] = {"open": open_price, "last": close_price}
    return quotes


def fetch_polygon_closing_quotes(tickers: list[str], as_of: date) -> dict[str, dict]:
    """One paid snapshot for post-bell display until the settled bar is stored."""
    key = polygon_api_key()
    if not key or not tickers:
        return {}
    return closing_quotes_from_polygon_snapshot(fetch_polygon_snapshot(key), set(tickers), as_of)


def snapshot_rejection_counts(raw_tickers: list[dict], wanted: set[str], as_of: date) -> dict[str, int]:
    """Explain why a nonempty paid snapshot did not fill the requested names."""
    matched = {}
    for item in raw_tickers:
        ticker = _matched_ticker(item, wanted)
        if ticker is not None:
            matched[ticker] = item
    counts: Counter[str] = Counter()
    for ticker in wanted:
        item = matched.get(ticker)
        if item is None:
            counts["absent"] += 1
            continue
        if _finite_positive((item.get("day") or {}).get("o")) is not None:
            counts["raw_day_open"] += 1
        quote, reason = _quote_and_rejection(item, as_of)
        counts["accepted" if quote is not None else reason or "unknown"] += 1
    return dict(counts)


def fetch_polygon_snapshot(api_key: str) -> list[dict]:
    """One full-market snapshot. Empty list on HTTP/JSON failure so the
    caller can fall through to Yahoo instead of aborting the scan."""
    response = None
    try:
        response = requests.get(
            POLYGON_SNAPSHOT_URL,
            params={"apiKey": api_key},
            timeout=SNAPSHOT_TIMEOUT_SECONDS,
        )
        if response.status_code == 403:
            from stock_picker.log import get_logger

            get_logger(__name__).warning(
                "polygon snapshot 403 -- key loaded, plan not entitled for day.o"
            )
            return []
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        from stock_picker.log import get_logger

        get_logger(__name__).warning(
            "polygon snapshot request failed http=%s",
            response.status_code if response is not None else None,
        )
        return []
    tickers = payload.get("tickers")
    if not isinstance(tickers, list):
        return []
    from stock_picker.log import get_logger

    get_logger(__name__).info(
        "polygon snapshot status=%s tickers=%s request_id=%s",
        payload.get("status"), len(tickers), payload.get("request_id"),
    )
    return tickers


def fetch_polygon_quotes(tickers: list[str], as_of: date, api_key: str | None = None) -> dict[str, dict]:
    """Today's open/last/prev_close for `tickers` from one Polygon snapshot.

    Missing key or a failed pull returns {} so fetch_quotes can use Yahoo.
    """
    key = api_key if api_key is not None else polygon_api_key()
    if not key or not tickers:
        return {}
    raw = fetch_polygon_snapshot(key)
    wanted = set(tickers)
    quotes = quotes_from_polygon_snapshot(raw, wanted, as_of)
    if len(quotes) < len(wanted) // 2:
        from stock_picker.log import get_logger

        samples = {
            ticker: {
                "updated_et": (
                    timestamp.isoformat()
                    if (timestamp := _eastern_datetime_from_ns(item.get("updated"))) else None
                ),
                "day_o": (item.get("day") or {}).get("o"),
                "day_c": (item.get("day") or {}).get("c"),
                "prev_day_c": (item.get("prevDay") or {}).get("c"),
                "last_trade_et": (
                    timestamp.isoformat()
                    if (timestamp := _eastern_datetime_from_ns((item.get("lastTrade") or {}).get("t"))) else None
                ),
            }
            for item in raw
            if (ticker := item.get("ticker")) in {"AAPL", "MSFT", "BMY"}
        }
        get_logger(__name__).warning(
            "polygon snapshot low coverage requested=%s reasons=%s samples=%s",
            len(wanted), snapshot_rejection_counts(raw, wanted, as_of), samples,
        )
    return quotes
