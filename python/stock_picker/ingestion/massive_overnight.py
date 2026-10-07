"""Verified raw Massive daily bars and corporate actions for overnight research.

No existing price/feature files are read or written here. A failed or partial
action request raises rather than certifying any date as action-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from stock_picker.ingestion.polygon_client import polygon_api_key

API_ROOT = "https://api.massive.com"
SPLITS_PATH = "/stocks/v1/splits"
DIVIDENDS_PATH = "/stocks/v1/dividends"
SOURCE = "massive"
ACTION_SOURCE = "massive_actions"
PRICE_BASIS = "raw"
REQUEST_TIMEOUT_SECONDS = 15
MAX_PAGES = 100
ACTION_LOOKBACK_DAYS = 700  # Inside the documented two-year Basic action history.
BAR_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]*\Z")


class MassiveOvernightError(ValueError):
    """A request or response could not establish complete raw/action evidence."""


@dataclass(frozen=True)
class ActionEvent:
    kind: str
    session: date
    event_id: str | None


@dataclass(frozen=True)
class VerifiedOvernightBars:
    history: pd.DataFrame
    provenance: pd.DataFrame
    split_events: tuple[ActionEvent, ...]
    dividend_events: tuple[ActionEvent, ...]


@dataclass(frozen=True)
class VerifiedCurrentOpen:
    """Session-dated raw open with action status checked by Massive."""

    open: float
    previous_close: float
    last_trade: float | None
    last_trade_at: datetime | None
    fetched_at: datetime
    corporate_action: str


def parse_current_snapshot(payload: object, ticker: str, session: date) -> tuple[float, float, float | None, datetime | None]:
    """Parse the documented single-ticker snapshot without trusting a stale row.

    https://massive.com/docs/rest/stocks/snapshots/single-ticker-snapshot
    The snapshot is a display/observed-open source, never an overnight label.
    """
    if not isinstance(payload, dict) or payload.get("status") != "OK":
        raise MassiveOvernightError("single-ticker snapshot was not OK")
    item = payload.get("ticker")
    if not isinstance(item, dict) or item.get("ticker") != ticker:
        raise MassiveOvernightError("single-ticker snapshot did not match the ticker")
    timestamp = item.get("updated") or (item.get("lastTrade") or {}).get("t")
    try:
        observed_at = datetime.fromtimestamp(int(timestamp) / 1_000_000_000, timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        raise MassiveOvernightError("single-ticker snapshot has no session timestamp") from exc
    if observed_at.date() != session:
        raise MassiveOvernightError("single-ticker snapshot is not dated to the scenario session")
    day = item.get("day") or {}
    previous = item.get("prevDay") or {}
    opened = _price(day.get("o"))
    previous_close = _price(previous.get("c"))
    trade = item.get("lastTrade") or {}
    trade_at = None
    trade_price = None
    if trade.get("t") is not None:
        try:
            trade_at = datetime.fromtimestamp(int(trade["t"]) / 1_000_000_000, timezone.utc).astimezone(
                ZoneInfo("America/New_York")
            )
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise MassiveOvernightError("single-ticker last-trade timestamp is invalid") from exc
        if trade_at.date() == session and trade.get("p") is not None:
            trade_price = _price(trade["p"])
        else:
            trade_at = None
    return opened, previous_close, trade_price, trade_at


def _price(value: object, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise MassiveOvernightError("invalid numeric bar field")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MassiveOvernightError("invalid numeric bar field") from exc
    if not isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise MassiveOvernightError("invalid numeric bar field")
    return number


def _event_date(value: object) -> date:
    if not isinstance(value, str):
        raise MassiveOvernightError("corporate action has no valid event date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise MassiveOvernightError("corporate action has no valid event date") from exc
    if parsed.isoformat() != value:
        raise MassiveOvernightError("corporate action has no valid event date")
    return parsed


def parse_daily_bar_page(payload: object, ticker: str, start: date, end: date) -> list[tuple[date, dict[str, float]]]:
    """Parse one `adjusted=false` aggregate response, rejecting mixed basis."""
    if not isinstance(payload, dict) or payload.get("status") != "OK":
        raise MassiveOvernightError("daily bars response was not OK")
    if payload.get("ticker") != ticker or payload.get("adjusted") is not False:
        raise MassiveOvernightError("daily bars ticker or raw basis was not confirmed")
    results = payload.get("results", [])
    if not isinstance(results, list) or payload.get("resultsCount") != len(results):
        raise MassiveOvernightError("daily bars result count was not confirmed")
    parsed = []
    for item in results:
        if not isinstance(item, dict) or isinstance(item.get("t"), bool) or not isinstance(item.get("t"), int):
            raise MassiveOvernightError("invalid daily bar timestamp")
        try:
            eastern = datetime.fromtimestamp(item["t"] / 1000, timezone.utc).astimezone(
                ZoneInfo("America/New_York")
            )
        except (ValueError, OverflowError, OSError) as exc:
            raise MassiveOvernightError("invalid daily bar timestamp") from exc
        if eastern.time().isoformat() != "00:00:00" or not start <= eastern.date() <= end:
            raise MassiveOvernightError("daily bar is outside the requested exchange-date window")
        bar = {
            "Open": _price(item.get("o")),
            "High": _price(item.get("h")),
            "Low": _price(item.get("l")),
            "Close": _price(item.get("c")),
            "Volume": _price(item.get("v"), allow_zero=True),
        }
        if bar["Low"] > min(bar["Open"], bar["Close"]) or bar["High"] < max(bar["Open"], bar["Close"]):
            raise MassiveOvernightError("daily bar OHLC values are inconsistent")
        parsed.append((eastern.date(), bar))
    return parsed


def parse_action_page(
    payload: object, ticker: str, start: date, end: date, *, kind: str
) -> list[ActionEvent]:
    """Parse one split/dividend page; missing dates cannot prove a clean day."""
    if kind not in {"split", "dividend"}:
        raise ValueError("unknown corporate action kind")
    if not isinstance(payload, dict) or payload.get("status") != "OK" or not isinstance(payload.get("results"), list):
        raise MassiveOvernightError(f"{kind} response was incomplete")
    date_field = "execution_date" if kind == "split" else "ex_dividend_date"
    events = []
    for item in payload["results"]:
        if not isinstance(item, dict) or item.get("ticker") != ticker:
            raise MassiveOvernightError(f"{kind} ticker was not confirmed")
        session = _event_date(item.get(date_field))
        if not start <= session <= end:
            raise MassiveOvernightError(f"{kind} event outside requested range")
        event_id = item.get("id")
        if event_id is not None and (not isinstance(event_id, str) or not event_id):
            raise MassiveOvernightError(f"{kind} event ID is invalid")
        events.append(ActionEvent(kind, session, event_id))
    return events


def _next_page_url(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise MassiveOvernightError("invalid pagination link")
    parsed = urlsplit(value)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    cursor_values = [val for key, val in query_pairs if key == "cursor"]
    if len(cursor_values) != 1 or not cursor_values[0] or any(
        key not in {"cursor", "apiKey"} for key, _ in query_pairs
    ):
        raise MassiveOvernightError("pagination link did not have a cursor-only scope")
    # Aggregate cursors may replace the range's `from` date with a Unix-ms
    # timestamp; keep the endpoint, ticker, cadence, and end date fixed.
    if path.startswith("/v2/aggs/ticker/"):
        expected = path.split("/")
        actual = parsed.path.split("/")
        same_path = (
            len(actual) == len(expected)
            and actual[:8] == expected[:8]
            and actual[-1] == expected[-1]
            and (actual[-2] == expected[-2] or actual[-2].isdigit())
        )
    else:
        same_path = parsed.path == path
    if parsed.scheme != "https" or parsed.netloc != "api.massive.com" or not same_path:
        raise MassiveOvernightError("pagination link left the expected Massive endpoint")
    # Some responses embed a key; authenticate each page ourselves and never
    # retain a provider-supplied key in the URL or error text.
    query = urlencode([(key, val) for key, val in query_pairs if key != "apiKey"])
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


class MassiveOvernightClient:
    """One ticker/range per call, with bounded complete pagination."""

    def __init__(
        self, api_key: str | None = None, *, session: requests.Session | None = None,
        max_pages: int = MAX_PAGES, action_coverage_start: date | None = None,
    ):
        self._api_key = api_key if api_key is not None else polygon_api_key()
        self._session = session or requests.Session()
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        self._max_pages = max_pages
        self._action_coverage_start = action_coverage_start

    def _get_payload(self, url: str, params: dict[str, object]) -> dict:
        if not self._api_key:
            raise MassiveOvernightError("Massive API key is unavailable")
        try:
            response = self._session.get(
                url, params={**params, "apiKey": self._api_key}, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException:
            raise MassiveOvernightError("Massive request failed") from None
        if response.status_code != 200:
            raise MassiveOvernightError(f"Massive HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise MassiveOvernightError("Massive response was not JSON") from exc
        if not isinstance(payload, dict):
            raise MassiveOvernightError("Massive response was not an object")
        return payload

    def _pages(self, path: str, params: dict[str, object]):
        url = API_ROOT + path
        seen: set[str] = set()
        for _ in range(self._max_pages):
            if url in seen:
                raise MassiveOvernightError("pagination cycle")
            seen.add(url)
            payload = self._get_payload(url, params)
            if url != API_ROOT + path and not payload.get("results"):
                raise MassiveOvernightError("empty cursor page cannot confirm pagination scope")
            yield payload
            following = _next_page_url(payload.get("next_url"), path)
            if following is None:
                return
            url = following
            params = {}
        raise MassiveOvernightError("Massive pagination exceeded the page limit")

    def _actions(self, path: str, field: str, kind: str, ticker: str, start: date, end: date) -> tuple[ActionEvent, ...]:
        """Bisect paginated date windows, certifying only filtered first pages.

        Massive cursors omit query filters and ignore any filters appended to
        them. A single-page request with explicit ticker/date/sort filters is
        the only page from which this method certifies action-free dates.
        """
        pending = [(start, end)]
        events: list[ActionEvent] = []
        requests_made = 0
        while pending:
            if requests_made >= self._max_pages:
                raise MassiveOvernightError("action verification exceeded the request limit")
            window_start, window_end = pending.pop()
            params = {
                "ticker": ticker,
                f"{field}.gte": window_start.isoformat(),
                f"{field}.lte": window_end.isoformat(),
                "sort": f"{field}.asc",
                "limit": 5000,
            }
            payload = self._get_payload(API_ROOT + path, params)
            requests_made += 1
            page_events = parse_action_page(payload, ticker, window_start, window_end, kind=kind)
            if [event.session for event in page_events] != sorted(event.session for event in page_events):
                raise MassiveOvernightError(f"{kind} events were not sorted by event date")
            if payload.get("next_url") is not None:
                _next_page_url(payload["next_url"], path)  # Validate; do not trust opaque cursor scope.
                if window_start == window_end:
                    raise MassiveOvernightError("one-day action window still requires pagination")
                midpoint = window_start + (window_end - window_start) // 2
                pending.extend([(midpoint + timedelta(days=1), window_end), (window_start, midpoint)])
                continue
            events.extend(page_events)
        return tuple(events)

    def fetch_current_open(self, ticker: str, session: date) -> VerifiedCurrentOpen:
        """Get a session-dated open; never certify an unchecked action day."""
        if not TICKER_PATTERN.fullmatch(ticker):
            raise ValueError("invalid ticker")
        snapshot = self._get_payload(
            API_ROOT + f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}", {}
        )
        opened, previous_close, last_trade, last_trade_at = parse_current_snapshot(snapshot, ticker, session)
        splits = self._actions(SPLITS_PATH, "execution_date", "split", ticker, session, session)
        dividends = self._actions(DIVIDENDS_PATH, "ex_dividend_date", "dividend", ticker, session, session)
        action = "split" if splits else "dividend" if dividends else "verified_none"
        return VerifiedCurrentOpen(
            opened, previous_close, last_trade, last_trade_at,
            datetime.now(timezone.utc), action,
        )

    def fetch(self, ticker: str, start: date, end: date) -> VerifiedOvernightBars:
        """Fetch all bars and action pages; certify only after every page succeeds."""
        if not TICKER_PATTERN.fullmatch(ticker) or start > end:
            raise ValueError("valid ticker and ordered date range required")
        if end >= datetime.now(ZoneInfo("America/New_York")).date():
            raise MassiveOvernightError("overnight bars require completed historical sessions")
        conservative_start = datetime.now(timezone.utc).date() - timedelta(days=ACTION_LOOKBACK_DAYS)
        coverage_start = max(conservative_start, self._action_coverage_start or conservative_start)
        if start < coverage_start:
            raise MassiveOvernightError("requested range predates verified action-feed coverage")
        bar_path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        bars: list[tuple[date, dict[str, float]]] = []
        for page in self._pages(bar_path, {"adjusted": "false", "sort": "asc", "limit": 50000}):
            bars.extend(parse_daily_bar_page(page, ticker, start, end))
        sessions = [day for day, _ in bars]
        if len(set(sessions)) != len(sessions) or sessions != sorted(sessions):
            raise MassiveOvernightError("daily bars had duplicate or unordered sessions")

        splits = self._actions(SPLITS_PATH, "execution_date", "split", ticker, start, end)
        dividends = self._actions(DIVIDENDS_PATH, "ex_dividend_date", "dividend", ticker, start, end)
        by_day: dict[date, list[ActionEvent]] = {}
        for event in (*splits, *dividends):
            by_day.setdefault(event.session, []).append(event)
        fetched_at = datetime.now(timezone.utc).isoformat()
        index = pd.DatetimeIndex(sessions, name="Date")
        history = pd.DataFrame([bar for _, bar in bars], index=index, columns=BAR_COLUMNS)
        provenance = pd.DataFrame(
            [
                {
                    "source": SOURCE,
                    "basis": PRICE_BASIS,
                    "action_source": ACTION_SOURCE,
                    "corporate_action": (
                        "split" if any(event.kind == "split" for event in by_day.get(day, ()))
                        else "dividend" if by_day.get(day) else "verified_none"
                    ),
                    "action_event_ids": "|".join(
                        f"{event.kind}:{event.event_id or 'unknown'}" for event in by_day.get(day, ())
                    ),
                    "action_verified_from": start.isoformat(),
                    "action_verified_to": end.isoformat(),
                    "actions_fetched_at": fetched_at,
                    "bars_fetched_at": fetched_at,
                }
                for day in sessions
            ],
            index=index,
        )
        return VerifiedOvernightBars(history, provenance, splits, dividends)
