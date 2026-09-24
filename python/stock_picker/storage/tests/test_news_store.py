from stock_picker.storage.news_store import NewsArticle, NewsStore


def _article(**overrides) -> NewsArticle:
    payload = dict(
        ticker="XENE",
        article_id="1",
        published_at="2026-09-22T08:01:00-04:00",
        as_of="2026-09-23",
        headline="Xenon pauses Phase 3 clinical trial after safety review",
        summary="Safety review halted the study.",
        source="Reuters",
        url="https://example.com/xene",
        material_score=0.91,
        is_material=1,
        phrase_hits=("clinical trial pause", "trial pause"),
    )
    payload.update(overrides)
    return NewsArticle(**payload)


def test_upsert_then_read_round_trips_an_article(tmp_path):
    store = NewsStore(data_dir=tmp_path)
    store.upsert([_article()])

    loaded = store.read()
    assert len(loaded) == 1
    assert loaded[0].ticker == "XENE"
    assert loaded[0].headline.startswith("Xenon pauses")
    assert loaded[0].is_material == 1
    assert loaded[0].phrase_hits == ("clinical trial pause", "trial pause")


def test_upsert_replaces_the_same_article_id(tmp_path):
    store = NewsStore(data_dir=tmp_path)
    store.upsert([_article(headline="old", material_score=0.2)])
    store.upsert([_article(headline="new", material_score=0.9)])

    loaded = store.read()
    assert len(loaded) == 1
    assert loaded[0].headline == "new"
    assert loaded[0].material_score == 0.9


def test_read_filters_by_ticker_and_as_of(tmp_path):
    store = NewsStore(data_dir=tmp_path)
    store.upsert(
        [
            _article(ticker="XENE", article_id="1"),
            _article(ticker="AAPL", article_id="1", headline="Apple event"),
            _article(ticker="XENE", article_id="2", as_of="2026-09-22", headline="older"),
        ]
    )

    assert [a.ticker for a in store.read(ticker="XENE", as_of="2026-09-23")] == ["XENE"]
    assert store.count(as_of="2026-09-23") == 2
    assert store.count() == 3


def test_mark_ingested_is_idempotent_per_ticker_day(tmp_path):
    store = NewsStore(data_dir=tmp_path)
    store.mark_ingested("XENE", "2026-09-23", article_count=2, ingested_at="2026-09-23T08:40:00")
    store.mark_ingested("XENE", "2026-09-23", article_count=0, ingested_at="2026-09-23T09:00:00")
    store.mark_ingested("AAPL", "2026-09-23", article_count=1, ingested_at="2026-09-23T09:01:00")

    assert store.tickers_ingested("2026-09-23") == {"XENE", "AAPL"}
    assert store.tickers_ingested("2026-09-22") == set()
