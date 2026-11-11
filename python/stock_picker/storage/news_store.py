"""Repository for universe company-news articles.

Same DI as TradeStore / PaperBookStore: `NewsStore(data_dir=tmp_path)`.
SQLite only -- one row per (ticker, article_id). Morning skip still uses
the short-list Finnhub path; this store is the training corpus (headline,
summary, material score, phrase hits).
"""

from __future__ import annotations

import sqlite3
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from stock_picker.storage.paths import data_root

DEFAULT_DATA_DIR = data_root() / "news"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_articles (
    ticker TEXT NOT NULL,
    article_id TEXT NOT NULL,
    published_at TEXT NOT NULL,
    as_of TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    url TEXT,
    material_score REAL NOT NULL,
    is_material INTEGER NOT NULL,
    phrase_hits TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (ticker, article_id)
);
CREATE INDEX IF NOT EXISTS idx_news_as_of ON news_articles(as_of);
CREATE INDEX IF NOT EXISTS idx_news_ticker_as_of ON news_articles(ticker, as_of);

CREATE TABLE IF NOT EXISTS news_ingest (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    article_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (ticker, as_of)
);

-- Forward observations only. Existing articles are deliberately not backdated.
CREATE TABLE IF NOT EXISTS news_fetch_observations (
    fetch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    requested_from TEXT NOT NULL,
    requested_through TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    classifier_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('success', 'error', 'unavailable')),
    error_kind TEXT,
    raw_json TEXT,
    valid_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS news_article_observations (
    fetch_id INTEGER NOT NULL REFERENCES news_fetch_observations(fetch_id),
    article_id TEXT NOT NULL,
    published_at TEXT NOT NULL,
    article_json TEXT NOT NULL,
    PRIMARY KEY(fetch_id, article_id)
);
CREATE INDEX IF NOT EXISTS idx_news_observed_ticker_time
    ON news_fetch_observations(ticker, observed_at);
"""


@dataclass(frozen=True)
class NewsArticle:
    ticker: str
    article_id: str
    published_at: str
    as_of: str
    headline: str
    summary: str | None = None
    source: str | None = None
    url: str | None = None
    material_score: float = 0.0
    is_material: int = 0
    phrase_hits: tuple[str, ...] = ()


def _hits_to_text(hits: tuple[str, ...]) -> str:
    return ",".join(hits)


def _hits_from_text(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part for part in raw.split(",") if part)


def _article_from_row(row: sqlite3.Row) -> NewsArticle:
    payload = dict(row)
    payload["phrase_hits"] = _hits_from_text(payload.get("phrase_hits"))
    return NewsArticle(**payload)


def _utc_text(value: str) -> str:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("news observation times require an explicit time zone")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="microseconds")


class NewsStore:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "news.db"
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def record_fetch(
        self, ticker: str, requested_from: str, requested_through: str,
        observed_at: str, raw: list[dict] | None, articles: list[NewsArticle],
        classifier_version: str, status: str = "success", error_kind: str | None = None,
    ) -> int:
        """Append raw coverage and immutable classified versions in one transaction."""
        timestamp = _utc_text(observed_at)
        unique = {article.article_id: article for article in articles}
        if any(article.ticker != ticker for article in unique.values()):
            raise ValueError("observation batch contains a different ticker")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO news_fetch_observations "
                "(ticker, requested_from, requested_through, observed_at, provider, classifier_version, "
                "status, error_kind, raw_json, valid_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, requested_from, requested_through, timestamp, "finnhub", classifier_version,
                 status, error_kind, json.dumps(raw, allow_nan=False) if raw is not None else None, len(unique)),
            )
            fetch_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO news_article_observations (fetch_id, article_id, published_at, article_json) "
                "VALUES (?, ?, ?, ?)",
                [(fetch_id, article.article_id, _utc_text(article.published_at),
                  json.dumps(asdict(article), allow_nan=False)) for article in unique.values()],
            )
        return fetch_id

    def observed_articles(self, ticker: str, at: str) -> list[dict]:
        """Latest version actually observed by the cutoff; never substitute legacy rows."""
        cutoff = _utc_text(at)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT a.article_id, a.article_json, f.observed_at, f.classifier_version "
                "FROM news_article_observations a JOIN news_fetch_observations f USING(fetch_id) "
                "WHERE f.ticker=? AND f.status='success' AND f.observed_at<=? AND a.published_at<=? "
                "ORDER BY f.observed_at DESC, f.fetch_id DESC", (ticker, cutoff, cutoff),
            ).fetchall()
        versions: dict[str, dict] = {}
        for row in rows:
            article_id = row["article_id"]
            if article_id not in versions:
                versions[article_id] = {"article": json.loads(row["article_json"]),
                                        "observed_at": row["observed_at"],
                                        "classifier_version": row["classifier_version"]}
            versions[article_id]["first_observed_at"] = row["observed_at"]
        return list(versions.values())

    def upsert(self, articles: list[NewsArticle]) -> None:
        if not articles:
            return
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO news_articles "
                "(ticker, article_id, published_at, as_of, headline, summary, "
                "source, url, material_score, is_material, phrase_hits) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        a.ticker,
                        a.article_id,
                        a.published_at,
                        a.as_of,
                        a.headline,
                        a.summary,
                        a.source,
                        a.url,
                        a.material_score,
                        a.is_material,
                        _hits_to_text(a.phrase_hits),
                    )
                    for a in articles
                ],
            )

    def mark_ingested(self, ticker: str, as_of: str, article_count: int, ingested_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO news_ingest "
                "(ticker, as_of, article_count, ingested_at) VALUES (?, ?, ?, ?)",
                (ticker, as_of, article_count, ingested_at),
            )

    def tickers_ingested(self, as_of: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker FROM news_ingest WHERE as_of = ?",
                (as_of,),
            ).fetchall()
        return {row["ticker"] for row in rows}

    def read(self, ticker: str | None = None, as_of: str | None = None) -> list[NewsArticle]:
        clauses: list[str] = []
        params: list[str] = []
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if as_of is not None:
            clauses.append("as_of = ?")
            params.append(as_of)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, article_id, published_at, as_of, headline, summary, "
                "source, url, material_score, is_material, phrase_hits "
                f"FROM news_articles {where} "
                "ORDER BY ticker, published_at, article_id",
                params,
            ).fetchall()
        return [_article_from_row(row) for row in rows]

    def count(self, as_of: str | None = None) -> int:
        if as_of is None:
            with self._connect() as conn:
                return int(conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0])
        with self._connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM news_articles WHERE as_of = ?",
                    (as_of,),
                ).fetchone()[0]
            )
