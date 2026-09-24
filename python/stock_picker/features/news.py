"""Company-news features from NewsStore.

Each history row is a completed session. Articles published that calendar
day (weekend prints roll back to Friday) land on that row. training/dataset
then shift(1)s them with the other close-known columns, so day t's session
sees yesterday's headlines -- never the same morning's. Empty store or no
articles yields zeros.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.storage.news_store import NewsArticle, NewsStore

EASTERN = ZoneInfo("America/New_York")
LOOKBACK_SESSIONS = 3
NEWS_FEATURE_COLUMNS = [
    "news_article_count_1d",
    "news_max_material_score_1d",
    "news_has_material_1d",
    "news_article_count_3d",
    "news_max_material_score_3d",
    "news_has_material_3d",
    "news_has_insider_sell_3d",
]
_INSIDER_PHRASES = frozenset(
    ("insider sell", "form 4", "cto sells", "cto sold", "sold shares")
)


def _session_date(value: str) -> date | None:
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        try:
            day = date.fromisoformat(value[:10])
        except ValueError:
            return None
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=EASTERN)
    day = stamp.astimezone(EASTERN).date()
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _prior_sessions(as_of: date, n: int) -> list[date]:
    """`as_of` plus n-1 prior weekdays, newest first."""
    out = [as_of]
    day = as_of
    while len(out) < n:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            out.append(day)
    return out


def _articles_on(articles: list[NewsArticle], day: date) -> list[NewsArticle]:
    return [article for article in articles if _session_date(article.published_at) == day]


def _count(articles: list[NewsArticle]) -> float:
    return float(len(articles))


def _max_score(articles: list[NewsArticle]) -> float:
    if not articles:
        return 0.0
    return float(max(article.material_score for article in articles))


def _has_material(articles: list[NewsArticle]) -> float:
    return 1.0 if any(article.is_material for article in articles) else 0.0


def _has_insider_sell(articles: list[NewsArticle]) -> float:
    for article in articles:
        hits = {phrase.lower() for phrase in article.phrase_hits}
        if hits.intersection(_INSIDER_PHRASES):
            return 1.0
    return 0.0


def build_news_features(
    history: pd.DataFrame,
    articles: list[NewsArticle] | None = None,
) -> pd.DataFrame:
    """One row per history date: that session's news plus a 3-session window."""
    if history.empty:
        return pd.DataFrame(columns=NEWS_FEATURE_COLUMNS)
    by_day = articles or []
    rows = []
    for stamp in history.index:
        as_of = stamp.date() if hasattr(stamp, "date") else pd.Timestamp(stamp).date()
        window_days = _prior_sessions(as_of, LOOKBACK_SESSIONS)
        last = _articles_on(by_day, as_of)
        window: list[NewsArticle] = []
        for day in window_days:
            window.extend(_articles_on(by_day, day))
        rows.append(
            {
                "news_article_count_1d": _count(last),
                "news_max_material_score_1d": _max_score(last),
                "news_has_material_1d": _has_material(last),
                "news_article_count_3d": _count(window),
                "news_max_material_score_3d": _max_score(window),
                "news_has_material_3d": _has_material(window),
                "news_has_insider_sell_3d": _has_insider_sell(window),
            }
        )
    return pd.DataFrame(rows, index=history.index)


def articles_by_ticker(store: NewsStore | None = None) -> dict[str, list[NewsArticle]]:
    news_store = store or NewsStore()
    grouped: dict[str, list[NewsArticle]] = {}
    for article in news_store.read():
        grouped.setdefault(article.ticker, []).append(article)
    return grouped
