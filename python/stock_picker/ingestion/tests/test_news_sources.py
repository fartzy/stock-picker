from datetime import date, datetime, timezone
from types import SimpleNamespace

import requests

from stock_picker.ingestion import news_sources as sources
from stock_picker.ingestion.finnhub_client import fetch_company_news

DAY = date(2026, 9, 30)
START = date(2026, 9, 29)
NOW = datetime(2026, 9, 30, 14, tzinfo=timezone.utc)


def article(title="A company headline", published="2026-09-30T00:04:08Z", url="https://example.com/news"):
    return {"headline": title, "summary": "Story details", "published_at": published, "url": url}


def test_secondary_source_is_fetched_even_when_primary_has_news(monkeypatch):
    monkeypatch.setattr(sources, "fetch_company_news", lambda *a: [article("Morning roundup")])
    monkeypatch.setattr(sources, "fetch_polygon_news", lambda *a: ([article("DEA judge pauses cannabis rescheduling", url="https://example.com/dea")], False))
    result = sources.fetch_news_coverage(["TRLV"], START, DAY)["TRLV"]
    assert len(result.articles) == 2
    assert result.sources == ["finnhub", "polygon"]
    assert result.issues == []


def test_empty_success_is_different_from_failed_feed():
    result = sources.merge_news_sources({"finnhub": [], "polygon": None}, START, DAY, NOW)
    assert result.sources == ["finnhub"]
    assert result.issues == ["polygon_unavailable"]
    assert not result.articles


def test_merge_deduplicates_and_enforces_publication_window():
    result = sources.merge_news_sources({"finnhub": [article()], "polygon": [
        article(url="https://example.com/news?tracking=2"),
        article("old", "2026-09-28T01:00:00Z"),
        article("future", "2026-09-30T21:00:00Z"),
        {"headline": "undated"},
    ]}, START, DAY, NOW)
    assert len(result.articles) == 1
    assert result.articles[0]["summary"] == "Story details"
    assert result.issues == ["polygon_undated_articles"]


def test_no_finnhub_key_is_a_failure_not_an_empty_feed(monkeypatch):
    monkeypatch.setattr("stock_picker.ingestion.finnhub_client.finnhub_api_key", lambda: None)
    assert fetch_company_news("TRLV", START, DAY) is None


def test_polygon_normalizes_articles_and_reports_pagination(monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "polygon_api_key", lambda: "dummy-key")
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "results": [{"title": "News", "published_utc": "2026-09-30T00:00:00Z", "description": "Summary", "publisher": {"name": "Publisher"}}],
            "next_url": "https://api.polygon.io/next",
        })
    monkeypatch.setattr(sources.requests, "get", get)
    rows, capped = sources.fetch_polygon_news("BRK-B", START, DAY)
    assert rows[0]["summary"] == "Summary"
    assert capped
    assert calls[0][1]["params"]["ticker"] == "BRK.B"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer dummy-key"}
    assert "apiKey" not in calls[0][1]["params"]


def test_polygon_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(sources, "polygon_api_key", lambda: "dummy-key")
    def fail(*args, **kwargs):
        raise requests.Timeout("fake provider timeout")
    monkeypatch.setattr(sources.requests, "get", fail)
    assert sources.fetch_polygon_news("TRLV", START, DAY) == (None, False)
