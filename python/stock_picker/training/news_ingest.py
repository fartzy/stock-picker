"""Full-universe Finnhub company-news ingest for training.

Morning skip still news-checks Fit + Rank 1-20 only. After that scan
releases its lock, this walks every active ticker, rate-limited to the
free-tier ceiling, and writes headline/summary plus the material
classifier (Tfidf + event phrases, including insider sells) into
NewsStore. Normal resume skips names already ingested for the date; explicit
refresh and the post-close candidate pass recheck them. Immutable observation
records preserve actual availability times and classifier versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_picker.ingestion.finnhub_client import (
    MIN_SECONDS_BETWEEN_CALLS,
    fetch_company_news,
    finnhub_api_key,
)
from stock_picker.log import get_logger
from stock_picker.storage.news_store import NewsArticle, NewsStore
from stock_picker.storage.paths import data_root
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.headline_sentiment import classify_article_text
from stock_picker.training import headline_sentiment

logger = get_logger(__name__)

LOOKBACK_SESSIONS = 3
EASTERN = ZoneInfo("America/New_York")


def lookback_start(as_of: date, sessions: int = LOOKBACK_SESSIONS) -> date:
    """First weekday included in the ingest window (sessions back from as_of)."""
    day = as_of
    seen = 0
    while seen < sessions:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            seen += 1
    return day


def _published_at_iso(payload: dict) -> str | None:
    raw = payload.get("datetime")
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(EASTERN).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _article_id(payload: dict) -> str | None:
    if payload.get("id") is not None and str(payload.get("id")).strip():
        return str(payload["id"]).strip()
    url = str(payload.get("url") or "").strip()
    if url:
        return url
    headline = str(payload.get("headline") or "").strip()
    return headline or None


def news_article_from_finnhub(ticker: str, payload: dict, as_of: date) -> NewsArticle | None:
    headline = str(payload.get("headline") or "").strip()
    article_id = _article_id(payload)
    published_at = _published_at_iso(payload)
    if not headline or not article_id or not published_at:
        return None
    summary = str(payload.get("summary") or "").strip() or None
    source = str(payload.get("source") or "").strip() or None
    url = str(payload.get("url") or "").strip() or None
    score, material, hits = classify_article_text(headline, summary or "")
    return NewsArticle(
        ticker=ticker,
        article_id=article_id,
        published_at=published_at,
        as_of=as_of.isoformat(),
        headline=headline,
        summary=summary,
        source=source,
        url=url,
        material_score=score,
        is_material=1 if material else 0,
        phrase_hits=tuple(hits),
    )


def ingest_universe_news(
    tickers: list[str] | None = None,
    as_of: date | None = None,
    store: NewsStore | None = None,
    fetch_one=fetch_company_news,
    sleep_seconds: float = MIN_SECONDS_BETWEEN_CALLS,
    lookback_sessions: int = LOOKBACK_SESSIONS,
    now: datetime | None = None,
    refresh: bool = False,
) -> dict[str, int]:
    """Rate-limited company-news for every active ticker.

    Empty result (no key, nothing pending) is a no-op, not an error.
    A ticker with zero articles is still marked ingested so a retry
    does not hammer Finnhub for silent names.
    """
    key = finnhub_api_key()
    names = tickers if tickers is not None else UniverseStore().active_tickers()
    day = as_of or date.today()
    news_store = store or NewsStore()
    if not key or not names:
        logger.info("universe news ingest skipped -- no key or no tickers")
        return {"tickers": 0, "articles": 0, "skipped": 0}
    already = news_store.tickers_ingested(day.isoformat())
    pending = list(dict.fromkeys(ticker for ticker in names if refresh or ticker not in already))
    start = lookback_start(day, lookback_sessions)
    classifier_version = (
        "headline-sha256:" + hashlib.sha256(Path(headline_sentiment.__file__).read_bytes()).hexdigest()
        + ";sklearn:" + version("scikit-learn")
    )
    def observed_now() -> str:
        return (now or datetime.now(EASTERN)).isoformat()
    n_tickers = 0
    n_articles = 0
    logger.info(
        "universe news ingest start tickers=%s pending=%s window=%s..%s",
        len(names),
        len(pending),
        start.isoformat(),
        day.isoformat(),
    )
    for index, ticker in enumerate(pending):
        if index and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            raw = fetch_one(ticker, start, day, api_key=key)
        except Exception as exc:
            logger.exception("universe news fetch failed ticker=%s", ticker)
            news_store.record_fetch(ticker, start.isoformat(), day.isoformat(), observed_now(),
                                    None, [], classifier_version, "error", type(exc).__name__)
            continue
        if raw is None:
            logger.warning("universe news fetch failed ticker=%s", ticker)
            news_store.record_fetch(ticker, start.isoformat(), day.isoformat(), observed_now(),
                                    None, [], classifier_version, "unavailable")
            continue
        articles = [
            article
            for row in raw
            if (article := news_article_from_finnhub(ticker, row, day)) is not None
        ]
        # Timestamp after fetching/classifying each ticker, not at batch start.
        ingested_at = observed_now()
        news_store.record_fetch(ticker, start.isoformat(), day.isoformat(), ingested_at,
                                raw, articles, classifier_version)
        news_store.upsert(articles)
        news_store.mark_ingested(
            ticker, day.isoformat(), article_count=len(articles), ingested_at=ingested_at
        )
        n_tickers += 1
        n_articles += len(articles)
        if n_tickers % 100 == 0:
            logger.info(
                "universe news progress %s/%s tickers %s articles",
                n_tickers,
                len(pending),
                n_articles,
            )
    logger.info(
        "universe news ingest done tickers=%s articles=%s skipped=%s",
        n_tickers,
        n_articles,
        len(names) - len(pending),
    )
    return {
        "tickers": n_tickers,
        "articles": n_articles,
        "skipped": len(names) - len(pending),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="recheck already-ingested names")
    parser.add_argument("--tickers", help="comma-separated ticker list; defaults to universe")
    args = parser.parse_args()
    names = [name.strip().upper() for name in args.tickers.split(",") if name.strip()] if args.tickers else None
    ingest_universe_news(tickers=names, refresh=args.refresh)


def refresh_candidate_news(as_of: date) -> dict[str, int]:
    """Recheck saved morning candidates at the close, including already-ingested names."""
    names = set()
    for suffix in ("", "-rank"):
        path = data_root() / "buy_signals" / f"{as_of.isoformat()}{suffix}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            if payload.get("as_of") != as_of.isoformat():
                raise ValueError("candidate session mismatch")
            names.update(row["ticker"] for row in payload.get("signals", [])[:20])
        except (OSError, ValueError, KeyError, TypeError):
            logger.exception("cannot read morning candidates from %s", path)
    return ingest_universe_news(tickers=sorted(names), as_of=as_of, refresh=True)


if __name__ == "__main__":
    main()
