"""Real-time Massive second aggregates for the regular-session opening price.

The stream's ``op`` is the official opening price; a second bar's ``o`` is
only that second's first trade. Start after the cash-market bell and retain
the REST snapshot/fallback path for names without a qualifying stream bar.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import date, datetime, time as clock_time, timezone
from zoneinfo import ZoneInfo

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from stock_picker.ingestion.polygon_client import fetch_polygon_snapshot, polygon_api_key
from stock_picker.ingestion.session import CHICAGO_TIMEZONE
from stock_picker.log import get_logger

logger = get_logger(__name__)

REALTIME_STOCKS_URL = "wss://socket.massive.com/stocks"
NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
REGULAR_OPEN = clock_time(9, 30)
STREAM_SUBSCRIPTION = "A.*"


def _positive_price(value: object) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def previous_closes_from_snapshot(raw: list[dict], wanted: set[str]) -> dict[str, float]:
    """Only use Massive's prior daily close, never a premarket last trade."""
    closes = {}
    for item in raw:
        symbol = item.get("ticker")
        if not isinstance(symbol, str):
            continue
        ticker = symbol if symbol in wanted else symbol.replace(".", "-")
        close = _positive_price((item.get("prevDay") or {}).get("c"))
        if ticker in wanted and close is not None:
            closes[ticker] = close
    return closes


def quote_from_second_aggregate(
    event: dict, as_of: date, previous_closes: dict[str, float]
) -> tuple[str, dict[str, float]] | None:
    """Accept only today's regular-session ``op`` from a second aggregate."""
    if event.get("ev") != "A":
        return None
    symbol = event.get("sym")
    if not isinstance(symbol, str):
        return None
    ticker = symbol if symbol in previous_closes else symbol.replace(".", "-")
    prev_close = previous_closes.get(ticker)
    if prev_close is None:
        return None
    try:
        started = datetime.fromtimestamp(int(event["s"]) / 1000, tz=timezone.utc).astimezone(
            NEW_YORK_TIMEZONE
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    if started.date() != as_of or started.time() < REGULAR_OPEN:
        return None
    open_price = _positive_price(event.get("op"))
    if open_price is None:
        return None
    last_price = _positive_price(event.get("c")) or open_price
    return ticker, {"open": open_price, "last": last_price, "prev_close": prev_close}


def _has_status(message: str | bytes, status: str) -> bool:
    events = json.loads(message)
    return isinstance(events, list) and any(
        isinstance(event, dict) and event.get("status") == status for event in events
    )


async def _collect_opens(
    key: str, wanted: set[str], as_of: date, deadline: float
) -> dict[str, dict[str, float]]:
    quotes: dict[str, dict[str, float]] = {}
    async with connect(REALTIME_STOCKS_URL, open_timeout=8, close_timeout=1, max_size=8_000_000) as socket:
        await asyncio.wait_for(socket.recv(), timeout=5)
        await socket.send(json.dumps({"action": "auth", "params": key}))
        if not _has_status(await asyncio.wait_for(socket.recv(), timeout=5), "auth_success"):
            logger.warning("polygon real-time stream authentication unavailable")
            return quotes
        await socket.send(json.dumps({"action": "subscribe", "params": STREAM_SUBSCRIPTION}))
        if not _has_status(await asyncio.wait_for(socket.recv(), timeout=5), "success"):
            logger.warning("polygon real-time aggregate subscription unavailable")
            return quotes

        previous_closes = previous_closes_from_snapshot(
            await asyncio.to_thread(fetch_polygon_snapshot, key), wanted
        )
        logger.info("polygon real-time aggregate subscription active prev_closes=%s", len(previous_closes))
        if not previous_closes:
            return quotes
        while time.monotonic() < deadline and len(quotes) < len(wanted):
            remaining = deadline - time.monotonic()
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                continue
            except ConnectionClosed:
                break
            try:
                events = json.loads(message)
            except ValueError:
                continue
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                parsed = quote_from_second_aggregate(event, as_of, previous_closes)
                if parsed is not None:
                    ticker, quote = parsed
                    quotes[ticker] = quote
    return quotes


def fetch_polygon_stream_quotes(
    tickers: list[str], as_of: date, cutoff: datetime
) -> dict[str, dict[str, float]]:
    """Collect official opens until cutoff; never block the fallback on a socket error."""
    key = polygon_api_key()
    remaining = (cutoff - datetime.now(CHICAGO_TIMEZONE)).total_seconds()
    if not key or not tickers or remaining <= 0:
        return {}
    try:
        quotes = asyncio.run(_collect_opens(key, set(tickers), as_of, time.monotonic() + remaining))
    except Exception as exc:
        # Transport errors can include a credential-bearing URL. Log the type only.
        logger.warning("polygon real-time aggregate stream failed: %s", type(exc).__name__)
        return {}
    logger.info("polygon real-time aggregate opens=%s/%s", len(quotes), len(set(tickers)))
    return quotes
