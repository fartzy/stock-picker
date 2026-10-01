"""Two independent shortlist feeds, with explicit failures and dated evidence.

Finnhub and Polygon/Massive use their existing local key readers. Query both
even if Finnhub returned articles: a nonempty feed can still miss a catalyst.
No network response or exception containing credentials is logged or stored.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests

from stock_picker.ingestion.finnhub_client import fetch_company_news
from stock_picker.ingestion.polygon_client import polygon_api_key

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
NEWS_TIMEOUT_SECONDS = 10
NEWS_FETCH_WORKERS = 8
MAX_NEWS_ARTICLES = 100
ET = ZoneInfo("America/New_York")


@dataclass
class NewsCoverage:
    articles: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def fetch_polygon_news(ticker: str, start: date, end: date) -> tuple[list[dict] | None, bool]:
    """None means unavailable; a second flag records capped/paginated results."""
    key = polygon_api_key()
    if not key:
        return None, False
    lower = datetime.combine(start, time.min, ET).astimezone(timezone.utc)
    upper = datetime.combine(end + timedelta(days=1), time.min, ET).astimezone(timezone.utc)
    try:
        response = requests.get(
            POLYGON_NEWS_URL,
            params={
                "ticker": ticker.replace("-", "."),
                "published_utc.gte": lower.isoformat(),
                "published_utc.lt": upper.isoformat(),
                "limit": MAX_NEWS_ARTICLES,
                "sort": "published_utc",
                "order": "desc",
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=NEWS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return None, False
        articles = []
        for row in payload["results"]:
            if not isinstance(row, dict):
                return None, False
            articles.append({
                "headline": row.get("title"),
                "summary": row.get("description"),
                "published_at": row.get("published_utc"),
                "url": row.get("article_url"),
                "source": (row.get("publisher") or {}).get("name"),
            })
        return articles, bool(payload.get("next_url"))
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return None, False


def _published_at(article: dict) -> datetime | None:
    try:
        if article.get("published_at"):
            value = datetime.fromisoformat(str(article["published_at"]).replace("Z", "+00:00"))
            return value.astimezone(timezone.utc) if value.tzinfo is not None else None
        return datetime.fromtimestamp(float(article["datetime"]), timezone.utc)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None


def merge_news_sources(
    feeds: dict[str, list[dict] | None], start: date, end: date, now: datetime | None = None,
) -> NewsCoverage:
    coverage = NewsCoverage()
    lower = datetime.combine(start, time.min, ET)
    upper = min(now or datetime.now(timezone.utc), datetime.combine(end + timedelta(days=1), time.min, ET))
    seen_headlines: set[str] = set()
    seen_urls: set[str] = set()
    for source, articles in feeds.items():
        if articles is None:
            coverage.issues.append(f"{source}_unavailable")
            continue
        coverage.sources.append(source)
        for article in articles:
            if not isinstance(article, dict):
                coverage.issues.append(f"{source}_invalid_articles")
                continue
            published = _published_at(article)
            if published is None:
                coverage.issues.append(f"{source}_undated_articles")
                continue
            if not lower <= published < upper:
                continue
            headline = str(article.get("headline") or "").strip()
            if not headline:
                coverage.issues.append(f"{source}_invalid_articles")
                continue
            url = str(article.get("url") or "").strip()
            try:
                parts = urlsplit(url)
                # Finnhub's public article links use /api/news?id=... . Only
                # tracking parameters are disposable; dropping id collapses
                # unrelated articles from the same endpoint into one story.
                query = urlencode([
                    (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                    if not key.lower().startswith("utm_") and key.lower() not in {"gclid", "fbclid"}
                ])
                url = urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))
            except ValueError:
                url = ""
            identity = re.sub(r"\W+", " ", headline).casefold().strip()
            if identity in seen_headlines or (url and url in seen_urls):
                continue
            seen_headlines.add(identity)
            if url:
                seen_urls.add(url)
            coverage.articles.append({
                "headline": headline[:500],
                "summary": str(article.get("summary") or "").strip()[:1500],
                "published_at": published.isoformat(),
                "url": url,
                "source": str(article.get("source") or source),
            })
    coverage.articles.sort(key=lambda row: row["published_at"], reverse=True)
    coverage.issues = list(dict.fromkeys(coverage.issues))
    return coverage


def fetch_news_coverage(tickers: list[str], start: date, end: date) -> dict[str, NewsCoverage]:
    names = list(dict.fromkeys(tickers))
    if not names:
        return {}

    def fetch_one(ticker: str) -> tuple[str, NewsCoverage]:
        try:
            primary = fetch_company_news(ticker, start, end)
        except (OSError, ValueError, TypeError):
            primary = None
        try:
            backup, truncated = fetch_polygon_news(ticker, start, end)
        except (OSError, ValueError, TypeError):
            backup, truncated = None, False
        coverage = merge_news_sources({"finnhub": primary, "polygon": backup}, start, end)
        if truncated:
            coverage.issues.append("polygon_truncated")
        return ticker, coverage

    with ThreadPoolExecutor(max_workers=min(NEWS_FETCH_WORKERS, len(names))) as pool:
        return dict(pool.map(fetch_one, names))
