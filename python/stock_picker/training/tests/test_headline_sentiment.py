from stock_picker.training.headline_sentiment import (
    is_material_negative,
    material_negative_score,
    news_flag_from_articles,
)


def test_trial_pause_scores_higher_than_ceo_pauses_to_thank_staff():
    trial = material_negative_score("Xenon pauses Phase 3 clinical trial after safety review")
    thanks = material_negative_score("CEO pauses to thank employees at conference")

    assert trial > thanks
    assert is_material_negative("Xenon pauses Phase 3 clinical trial after safety review")
    assert not is_material_negative("CEO pauses to thank employees at conference")


def test_news_flag_from_articles_picks_the_trial_headline():
    flag = news_flag_from_articles(
        [
            {"headline": "CEO pauses to thank employees at conference", "summary": ""},
            {"headline": "Xenon pauses Phase 3 clinical trial after safety review", "summary": ""},
        ]
    )

    assert flag is not None
    assert "clinical trial" in flag.lower()
