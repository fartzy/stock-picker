from stock_picker.training.news_day_judge import (
    _parse_judge,
    flag_from_articles,
    grok_judge,
    llm_chat_url,
    llm_model,
)


def test_llm_chat_url_uses_proxy_env(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_BASE_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("LLM_NEWS_MODEL", "local-test-model")
    assert llm_chat_url() == "http://127.0.0.1:4000/v1/chat/completions"
    assert llm_model() == "local-test-model"


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

def test_flag_from_articles_skips_analyst_initiate_when_grok_is_none(monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.grok_judge",
        lambda ticker, headlines, api_key=None: None,
    )
    flag = flag_from_articles(
        "HUT",
        [{"headline": "Wells Fargo Initiates Coverage On Hut 8"}],
    )
    assert flag is None


def test_flag_from_articles_empty_is_trust(monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.grok_judge",
        lambda ticker, headlines, api_key=None: None,
    )
    assert flag_from_articles("BVC", []) is None
