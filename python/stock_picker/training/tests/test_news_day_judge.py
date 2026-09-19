from stock_picker.training.news_day_judge import _parse_judge, flag_from_articles, grok_judge


def test_parse_judge_avoid_true():
    assert _parse_judge('{"avoid": true, "reason": "Phase 3 trial pause"}') == (
        True,
        "Phase 3 trial pause",
    )


def test_parse_judge_trust_the_tape():
    assert _parse_judge('{"avoid": false, "reason": ""}') == (False, "")


def test_parse_judge_ignores_garbage():
    assert _parse_judge("not json") is None


def test_flag_from_articles_uses_grok_when_it_returns_avoid(monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.grok_judge",
        lambda ticker, headlines, api_key=None: (True, "Phase 3 trial pause"),
    )
    flag = flag_from_articles(
        "XENE",
        [{"headline": "Xenon pauses Phase 3 clinical trial after safety review"}],
    )
    assert flag == "Phase 3 trial pause"


def test_flag_from_articles_falls_back_to_sklearn_when_grok_is_none(monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.grok_judge",
        lambda ticker, headlines, api_key=None: None,
    )
    flag = flag_from_articles(
        "XENE",
        [{"headline": "Xenon pauses Phase 3 clinical trial after safety review"}],
    )
    assert flag is not None
    assert "trial" in flag.lower()
