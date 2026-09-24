from datetime import date, datetime
from zoneinfo import ZoneInfo

from stock_picker.storage.news_store import NewsStore
from stock_picker.training.news_ingest import (
    ingest_universe_news,
    lookback_start,
    news_article_from_finnhub,
)


def _unix(year, month, day, hour=8, minute=1):
    dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    return int(dt.timestamp())


def test_lookback_start_skips_the_weekend():
    assert lookback_start(date(2026, 9, 23), sessions=3) == date(2026, 9, 18)


def test_news_article_from_finnhub_stores_classifier_fields():
    payload = {
        "id": 99,
        "headline": "Fastly's CTO Sells Over 33,000 Shares for $825,000 After the Stock Rose",
        "summary": "Form 4 filing after the rally.",
        "source": "Benzinga",
        "url": "https://example.com/fsly",
        "datetime": _unix(2026, 9, 22),
    }

    article = news_article_from_finnhub("FSLY", payload, date(2026, 9, 23))

    assert article is not None
    assert article.ticker == "FSLY"
    assert article.is_material == 1
    assert article.material_score > 0.5
    assert "cto sells" in article.phrase_hits
    assert article.summary.startswith("Form 4")


def test_ingest_universe_news_writes_articles_and_is_resume_safe(tmp_path, monkeypatch):
    store = NewsStore(data_dir=tmp_path)
    calls = []

    def fetch_one(ticker, start, as_of, api_key=None):
        calls.append(ticker)
        if ticker == "XENE":
            return [
                {
                    "id": "1",
                    "headline": "Xenon pauses Phase 3 clinical trial after safety review",
                    "summary": "",
                    "source": "Reuters",
                    "url": "https://example.com/xene",
                    "datetime": _unix(2026, 9, 22),
                }
            ]
        return []

    monkeypatch.setattr("stock_picker.training.news_ingest.finnhub_api_key", lambda: "k")

    first = ingest_universe_news(
        tickers=["XENE", "AAPL"],
        as_of=date(2026, 9, 23),
        store=store,
        fetch_one=fetch_one,
        sleep_seconds=0,
    )
    second = ingest_universe_news(
        tickers=["XENE", "AAPL"],
        as_of=date(2026, 9, 23),
        store=store,
        fetch_one=fetch_one,
        sleep_seconds=0,
    )

    assert first == {"tickers": 2, "articles": 1, "skipped": 0}
    assert second == {"tickers": 0, "articles": 0, "skipped": 2}
    assert calls == ["XENE", "AAPL"]
    loaded = store.read(ticker="XENE")
    assert len(loaded) == 1
    assert loaded[0].is_material == 1
    assert store.tickers_ingested("2026-09-23") == {"XENE", "AAPL"}


def test_ingest_universe_news_is_a_noop_without_an_api_key(tmp_path, monkeypatch):
    store = NewsStore(data_dir=tmp_path)
    monkeypatch.setattr("stock_picker.training.news_ingest.finnhub_api_key", lambda: None)
    called = []

    stats = ingest_universe_news(
        tickers=["XENE"],
        as_of=date(2026, 9, 23),
        store=store,
        fetch_one=lambda *a, **k: called.append(True) or [],
        sleep_seconds=0,
    )

    assert stats == {"tickers": 0, "articles": 0, "skipped": 0}
    assert called == []
    assert store.read() == []



def test_failed_ticker_is_not_marked_ingested_so_a_retry_can_fill_it(tmp_path, monkeypatch):
    store = NewsStore(data_dir=tmp_path)
    monkeypatch.setattr("stock_picker.training.news_ingest.finnhub_api_key", lambda: "k")

    def fetch_one(ticker, start, as_of, api_key=None):
        if ticker == "BAD":
            raise RuntimeError("429")
        return []

    ingest_universe_news(
        tickers=["BAD", "AAPL"],
        as_of=date(2026, 9, 23),
        store=store,
        fetch_one=fetch_one,
        sleep_seconds=0,
    )

    assert store.tickers_ingested("2026-09-23") == {"AAPL"}



def test_none_fetch_is_not_marked_ingested(tmp_path, monkeypatch):
    store = NewsStore(data_dir=tmp_path)
    monkeypatch.setattr("stock_picker.training.news_ingest.finnhub_api_key", lambda: "k")

    def fetch_one(ticker, start, as_of, api_key=None):
        if ticker == "BAD":
            return None
        return []

    ingest_universe_news(
        tickers=["BAD", "AAPL"],
        as_of=date(2026, 9, 23),
        store=store,
        fetch_one=fetch_one,
        sleep_seconds=0,
    )

    assert store.tickers_ingested("2026-09-23") == {"AAPL"}
