from stock_picker.training.headline_sentiment import (
    classify_article_text,
    is_material_news,
    is_material_negative,
    material_negative_score,
    material_phrase_hits,
    news_flag_from_articles,
)


def test_trial_pause_scores_higher_than_ceo_pauses_to_thank_staff():
    trial = material_negative_score("Xenon pauses Phase 3 clinical trial after safety review")
    thanks = material_negative_score("CEO pauses to thank employees at conference")

    assert trial > thanks
    assert is_material_negative("Xenon pauses Phase 3 clinical trial after safety review")
    assert not is_material_negative("CEO pauses to thank employees at conference")


def test_buyout_is_material_and_analyst_initiate_is_not():
    assert is_material_news("Company to be acquired at a 40 percent premium")
    assert not is_material_news("Wells Fargo initiates coverage with overweight rating")


def test_news_flag_from_articles_picks_the_trial_headline():
    flag = news_flag_from_articles(
        [
            {"headline": "CEO pauses to thank employees at conference", "summary": ""},
            {"headline": "Xenon pauses Phase 3 clinical trial after safety review", "summary": ""},
        ]
    )

    assert flag is not None
    assert "clinical trial" in flag.lower()


def test_classify_article_text_exposes_score_flag_and_phrase_hits():
    score, material, hits = classify_article_text(
        "Insider sold 40,000 shares in Form 4 filing",
        "",
    )

    assert material
    assert score > 0.5
    assert "form 4" in hits
    assert "insider sell" not in material_phrase_hits("Analyst upgrades shares to buy on growth")
