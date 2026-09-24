import pandas as pd

from stock_picker.features.news import NEWS_FEATURE_COLUMNS, build_news_features
from stock_picker.storage.news_store import NewsArticle


def _history():
    return pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2026-09-21", "2026-09-22", "2026-09-23"], name="date"),
    )


def _article(published_at, **overrides):
    payload = dict(
        ticker="XENE",
        article_id="1",
        published_at=published_at,
        as_of="2026-09-23",
        headline="Xenon pauses Phase 3 clinical trial after safety review",
        summary="",
        source="Reuters",
        url="https://example.com/xene",
        material_score=0.91,
        is_material=1,
        phrase_hits=("clinical trial pause",),
    )
    payload.update(overrides)
    return NewsArticle(**payload)


def test_news_features_are_zero_when_there_are_no_articles():
    features = build_news_features(_history(), articles=[])

    assert list(features.columns) == NEWS_FEATURE_COLUMNS
    assert (features == 0.0).all().all()


def test_news_features_land_on_the_article_session_date():
    articles = [
        _article("2026-09-22T16:01:00-04:00", article_id="after-close"),
        _article("2026-09-23T08:01:00-04:00", article_id="next-morning"),
    ]

    features = build_news_features(_history(), articles=articles)
    tuesday = features.loc[pd.Timestamp("2026-09-22")]
    wednesday = features.loc[pd.Timestamp("2026-09-23")]

    assert tuesday["news_article_count_1d"] == 1.0
    assert tuesday["news_has_material_1d"] == 1.0
    assert tuesday["news_max_material_score_1d"] == 0.91
    assert wednesday["news_article_count_1d"] == 1.0
    assert wednesday["news_article_count_3d"] == 2.0


def test_insider_sell_flag_covers_three_sessions_including_today():
    articles = [
        _article(
            "2026-09-21T10:00:00-04:00",
            article_id="form4",
            headline="Insider sold 40,000 shares in Form 4 filing",
            material_score=0.8,
            is_material=1,
            phrase_hits=("form 4", "sold shares"),
        )
    ]

    features = build_news_features(_history(), articles=articles)
    monday = features.loc[pd.Timestamp("2026-09-21")]
    wednesday = features.loc[pd.Timestamp("2026-09-23")]

    assert monday["news_has_insider_sell_3d"] == 1.0
    assert monday["news_article_count_1d"] == 1.0
    assert wednesday["news_has_insider_sell_3d"] == 1.0
    assert wednesday["news_article_count_1d"] == 0.0
