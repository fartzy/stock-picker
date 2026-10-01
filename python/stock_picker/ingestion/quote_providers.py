"""Today's session quotes as a chain of providers.

Morning scoring needs `{ticker: {open, last, prev_close}}` dated `as_of`.
Each provider fills names the previous one missed. Default order:

  polygon  -- REST snapshot `day.o` (opening cross). One call, whole tape.
  yahoo    -- 9:30 ET 1-minute print, then snapshot, never yesterday's Open.
  finnhub  -- leftover only.

Swap with `QUOTE_PROVIDERS=polygon,yahoo,finnhub`. Unknown names are skipped.
A provider that returns {} (no key, 403, network) is not an error -- the
next one runs.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, time as clock_time, timedelta
from typing import Callable, Protocol, TypedDict

from stock_picker.ingestion.session import CHICAGO_TIMEZONE
from stock_picker.log import get_logger

logger = get_logger(__name__)

QUOTE_PROVIDERS_ENV = "QUOTE_PROVIDERS"
DEFAULT_PROVIDER_NAMES = ("polygon", "yahoo", "finnhub")
MORNING_FIRST_SNAPSHOT = clock_time(8, 30, 5)
MORNING_SNAPSHOT_CUTOFF = clock_time(8, 33)
MORNING_RETRY_SECONDS = 5


class SessionQuote(TypedDict):
    open: float
    last: float
    prev_close: float


class QuoteProvider(Protocol):
    name: str

    def fetch(self, tickers: list[str], as_of: date) -> dict[str, SessionQuote]:
        """Today's open/last/prev_close. Omit names you cannot fill."""


class PolygonQuoteProvider:
    """Official regular-session open from Polygon `day.o`.

    Opening auctions can print after the bell. Names with no `day.o` yet
    are omitted -- we do not invent an open from last trade.
    """

    name = "polygon"

    def fetch(self, tickers: list[str], as_of: date) -> dict[str, SessionQuote]:
        from stock_picker.ingestion.polygon_client import fetch_polygon_quotes

        return fetch_polygon_quotes(tickers, as_of=as_of)


class YahooQuoteProvider:
    """Free path: 1-minute RTH open, then v7 snapshot, then daily.

    Drops a snapshot open that is yesterday's official Open to the cent
    (WRBY $23.18). That rule is Yahoo-only -- Polygon `day.o` is already
    today's print, even when it equals yesterday's Open.
    """

    name = "yahoo"

    def fetch(self, tickers: list[str], as_of: date) -> dict[str, SessionQuote]:
        from stock_picker.ingestion.yfinance_client import (
            drop_copied_prior_opens,
            fetch_yahoo_quotes,
            prior_session_opens,
        )

        quotes = fetch_yahoo_quotes(tickers, as_of=as_of)
        prior = prior_session_opens(tickers, as_of)
        if prior:
            quotes = drop_copied_prior_opens(quotes, prior)
        return quotes


class FinnhubQuoteProvider:
    name = "finnhub"

    def fetch(self, tickers: list[str], as_of: date) -> dict[str, SessionQuote]:
        from stock_picker.ingestion.finnhub_client import fetch_finnhub_quotes

        return fetch_finnhub_quotes(tickers, as_of=as_of)


_REGISTRY = {
    "polygon": PolygonQuoteProvider,
    "yahoo": YahooQuoteProvider,
    "finnhub": FinnhubQuoteProvider,
}


def provider_names_from_env(raw: str | None = None) -> tuple[str, ...]:
    text = (raw if raw is not None else os.environ.get(QUOTE_PROVIDERS_ENV, "")).strip()
    if not text:
        return DEFAULT_PROVIDER_NAMES
    names = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    return names or DEFAULT_PROVIDER_NAMES


def quote_providers(names: tuple[str, ...] | None = None) -> list[QuoteProvider]:
    chosen = names if names is not None else provider_names_from_env()
    providers: list[QuoteProvider] = []
    for name in chosen:
        cls = _REGISTRY.get(name)
        if cls is None:
            logger.warning("unknown quote provider %s -- skip", name)
            continue
        providers.append(cls())
    return providers


def fill_quotes(
    tickers: list[str],
    as_of: date,
    providers: list[QuoteProvider],
) -> dict[str, SessionQuote]:
    """Walk providers until every ticker has a today open, or the chain ends."""
    quotes: dict[str, SessionQuote] = {}
    missing = list(tickers)
    for provider in providers:
        if not missing:
            break
        batch = provider.fetch(missing, as_of)
        quotes.update(batch)
        missing = [ticker for ticker in tickers if ticker not in quotes]
        logger.info(
            "quotes %s filled=%s still_missing=%s",
            provider.name,
            len(batch),
            len(missing),
        )
    return quotes


def fetch_quotes(
    tickers: list[str],
    as_of: date | None = None,
    providers: list[QuoteProvider] | None = None,
) -> dict[str, SessionQuote]:
    """Public facade. Morning scoring and /api/quotes both come through here."""
    as_of = as_of or date.today()
    chain = providers if providers is not None else quote_providers()
    return fill_quotes(tickers, as_of, chain)


def wait_for_morning_snapshot(
    *,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until 08:30:05 CT; safe to call after that time as well."""
    now = now_fn or (lambda: datetime.now(CHICAGO_TIMEZONE))
    current = now()
    first_at = datetime.combine(current.date(), MORNING_FIRST_SNAPSHOT, CHICAGO_TIMEZONE)
    remaining = (first_at - current).total_seconds()
    if remaining > 0:
        sleep_fn(remaining)


def fetch_morning_quotes(
    tickers: list[str],
    as_of: date | None = None,
    providers: list[QuoteProvider] | None = None,
    *,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, SessionQuote]:
    """Retry the full-market Polygon snapshot for the scheduled/manual scan.

    The first request is no earlier than 08:30:05 Chicago time. Keep the
    names found on earlier pulls, retry missing names every five seconds
    through 08:33, then run the remaining providers once. Other quote API
    callers keep the ordinary single-pass provider chain.
    """
    now = now_fn or (lambda: datetime.now(CHICAGO_TIMEZONE))
    as_of = as_of or now().date()
    names = list(dict.fromkeys(tickers))
    chain = providers if providers is not None else quote_providers()
    if not names or not chain:
        return {}
    if chain[0].name != "polygon" or as_of != now().date():
        return fill_quotes(names, as_of, chain)

    if providers is None:
        from stock_picker.ingestion.polygon_client import polygon_api_key

        if not polygon_api_key():
            logger.warning("polygon key unavailable -- using fallback providers")
            return fill_quotes(names, as_of, chain[1:])

    first_at = datetime.combine(as_of, MORNING_FIRST_SNAPSHOT, CHICAGO_TIMEZONE)
    cutoff = datetime.combine(as_of, MORNING_SNAPSHOT_CUTOFF, CHICAGO_TIMEZONE)
    next_pull = max(first_at, now())
    quotes: dict[str, SessionQuote] = {}
    attempts = 0
    while True:
        remaining = (next_pull - now()).total_seconds()
        if remaining > 0:
            sleep_fn(remaining)
        missing = [ticker for ticker in names if ticker not in quotes]
        batch = chain[0].fetch(missing, as_of)
        quotes.update(batch)
        attempts += 1
        logger.info(
            "morning polygon attempt=%s filled=%s/%s still_missing=%s",
            attempts,
            len(quotes),
            len(names),
            len(names) - len(quotes),
        )
        if len(quotes) == len(names) or now() >= cutoff:
            break
        next_pull += timedelta(seconds=MORNING_RETRY_SECONDS)
        while next_pull <= now():
            next_pull += timedelta(seconds=MORNING_RETRY_SECONDS)
        if next_pull > cutoff:
            break

    missing = [ticker for ticker in names if ticker not in quotes]
    if missing:
        quotes.update(fill_quotes(missing, as_of, chain[1:]))
    return quotes
