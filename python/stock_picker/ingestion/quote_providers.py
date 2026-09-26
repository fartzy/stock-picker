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
from datetime import date
from typing import Protocol, TypedDict

from stock_picker.log import get_logger

logger = get_logger(__name__)

QUOTE_PROVIDERS_ENV = "QUOTE_PROVIDERS"
DEFAULT_PROVIDER_NAMES = ("polygon", "yahoo", "finnhub")


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

    Empty until the opening cross prints (~8:30:01–8:30:03 CT). Illiquid
    names with no `day.o` yet are omitted -- we do not invent an open from
    last trade.
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
