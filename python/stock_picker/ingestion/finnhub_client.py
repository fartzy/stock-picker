"""Finnhub /quote as a leftover filler for this-morning opens.

Free Finnhub is one ticker per request and ~60 calls/minute, so this is
never the bulk path for ~2000 names -- Yahoo's snapshot (then 1-minute
bars) does that. After those, any still-missing names get `/quote`, which
returns today's `o` (open), `c` (last), and `pc` (previous close) in one
JSON object. Dated by `t` (unix seconds of the last update): yesterday's
quote is dropped, same as every other source.

Key lookup, first hit wins: `FINNHUB_API_KEY`, then
`~/.config/api/finnhub.txt` (`email: key`, `FINNHUB_API_KEY=`, or a bare
key).
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_EARNINGS_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/earnings"
FINNHUB_API_KEY_ENV = "FINNHUB_API_KEY"
FINNHUB_KEY_FILE = Path.home() / ".config" / "api" / "finnhub.txt"
QUOTE_TIMEOUT_SECONDS = 10
# Free-tier ceiling. Stay just under it so a leftover fill doesn't 429.
MIN_SECONDS_BETWEEN_CALLS = 1.05
# 200 leftover names would take ~3.5 minutes -- skip Finnhub and leave them
# missing rather than blocking the whole morning scan.
MAX_LEFTOVER_TICKERS = 40


def _assigned_value(line: str, name: str) -> str | None:
    prefix = f"{name}="
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    value = stripped[len(prefix) :].strip().strip("'\"")
    return value or None


def _key_from_config_file(path: Path) -> str | None:
    """`email: key`, `FINNHUB_API_KEY=...`, or a file whose only non-comment
    line is the key itself."""
    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    for line in lines:
        value = _assigned_value(line, FINNHUB_API_KEY_ENV)
        if value:
            return value
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if ":" in stripped and not stripped.startswith("http"):
            _, _, rest = stripped.partition(":")
            token = rest.strip().strip("'\"")
            if token:
                return token
    nonempty = [line.strip().strip("'\"") for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(nonempty) == 1 and "=" not in nonempty[0] and ":" not in nonempty[0]:
        return nonempty[0]
    return None


def finnhub_api_key(key_file: Path | None = None) -> str | None:
    key = os.environ.get(FINNHUB_API_KEY_ENV, "").strip()
    if key:
        return key
    return _key_from_config_file(key_file or FINNHUB_KEY_FILE)


def finnhub_symbol(ticker: str) -> str:
    """Our universe uses Yahoo-style class shares (BRK-B); Finnhub wants BRK.B."""
    return ticker.replace("-", ".")


def _session_date_from_unix(timestamp: object) -> date | None:
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
    if number != number or number <= 0:
        return None
    return number


def quote_from_finnhub_payload(payload: dict, as_of: date) -> dict | None:
    """Shape one Finnhub /quote JSON object. `o` is the day's open -- not
    `c` (last). Stale (`t` not today) or missing open is dropped."""
    if _session_date_from_unix(payload.get("t")) != as_of:
        return None
    open_price = _finite_positive(payload.get("o"))
    last_price = _finite_positive(payload.get("c"))
    prev_close = _finite_positive(payload.get("pc"))
    if open_price is None or last_price is None or prev_close is None:
        return None
    return {"open": open_price, "last": last_price, "prev_close": prev_close}


def fetch_one_finnhub_quote(ticker: str, api_key: str, session: requests.Session | None = None) -> dict | None:
    client = session or requests
    try:
        response = client.get(
            FINNHUB_QUOTE_URL,
            params={"symbol": finnhub_symbol(ticker), "token": api_key},
            timeout=QUOTE_TIMEOUT_SECONDS,
        )
        if response.status_code == 429:
            return None
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def fetch_finnhub_quotes(
    tickers: list[str],
    as_of: date,
    api_key: str | None = None,
    sleep_seconds: float = MIN_SECONDS_BETWEEN_CALLS,
) -> dict[str, dict]:
    """Today's open/last/prev_close for leftover tickers. Empty if no key,
    nothing left, or more leftovers than MAX_LEFTOVER_TICKERS."""
    key = api_key if api_key is not None else finnhub_api_key()
    if not key or not tickers or len(tickers) > MAX_LEFTOVER_TICKERS:
        return {}
    quotes = {}
    session = requests.Session()
    for index, ticker in enumerate(tickers):
        if index and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        payload = fetch_one_finnhub_quote(ticker, key, session=session)
        if not payload:
            continue
        quote = quote_from_finnhub_payload(payload, as_of)
        if quote:
            quotes[ticker] = quote
    return quotes


def earnings_tickers_from_calendar(payload: dict, wanted: set[str], as_of: date) -> set[str]:
    """Tickers in `wanted` with an earnings date of `as_of`.

    Finnhub's calendar `date` is the report date -- BMO names print into
    today's open, AMC names from yesterday already moved overnight. Callers
    that want both should union as_of with the prior session. Pure, no network.
    """
    hits = set()
    as_of_text = as_of.isoformat()
    for row in payload.get("earningsCalendar") or []:
        symbol = row.get("symbol")
        if not symbol or row.get("date") != as_of_text:
            continue
        ticker = symbol.replace(".", "-")
        if ticker in wanted or symbol in wanted:
            hits.add(ticker if ticker in wanted else symbol)
    return hits


def fetch_earnings_tickers(
    tickers: list[str],
    as_of: date,
    from_date: date | None = None,
    api_key: str | None = None,
) -> set[str]:
    """Universe names with earnings on `from_date`..`as_of` (inclusive).

    One calendar request, not 2000. Empty set if no key or the pull fails --
    live scoring then proceeds without an earnings skip.
    """
    key = api_key if api_key is not None else finnhub_api_key()
    if not key or not tickers:
        return set()
    start = from_date or as_of
    try:
        response = requests.get(
            FINNHUB_EARNINGS_CALENDAR_URL,
            params={"from": start.isoformat(), "to": as_of.isoformat(), "token": key},
            timeout=QUOTE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    wanted = set(tickers)
    hits: set[str] = set()
    day = start
    while day <= as_of:
        hits |= earnings_tickers_from_calendar(payload, wanted, day)
        day += timedelta(days=1)
    return hits
