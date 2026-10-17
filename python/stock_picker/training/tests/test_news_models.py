"""News model status must count real decisions, not invent accuracy."""

from stock_picker.training import news_models


def scan(*checks):
    return {"signals": [{"ticker": f"T{i}", "news_check": check} for i, check in enumerate(checks)]}


def test_counts_decisions_and_flags_by_judge():
    usage = news_models.summarize_news_checks([
        scan(
            {"judge": "grok", "flag": None, "status": "complete"},
            {"judge": "grok", "flag": "takeover", "status": "complete"},
        ),
        scan(
            {"judge": "classifier", "flag": None, "status": "degraded", "issues": ["llm_unavailable"]},
            {"judge": "event_guard", "flag": "corporate action", "status": "complete"},
        ),
    ])
    assert usage["checked"] == 4
    assert usage["flagged"] == 2
    assert usage["by_judge"]["grok"] == {"decided": 2, "flagged": 1}
    assert usage["by_judge"]["event_guard"] == {"decided": 1, "flagged": 1}
    assert usage["by_status"] == {"complete": 3, "degraded": 1}
    assert usage["llm_unavailable"] == 1


def test_signals_without_a_news_check_are_skipped():
    usage = news_models.summarize_news_checks([{"signals": [{"ticker": "A"}]}, {}])
    assert usage == {"checked": 0, "flagged": 0, "by_judge": {}, "by_status": {}, "llm_unavailable": 0}


def test_status_lists_the_three_deciders_in_run_order(monkeypatch):
    monkeypatch.setattr(news_models, "llm_api_key", lambda: None)
    monkeypatch.setattr(news_models, "llm_chat_url", lambda: news_models.XAI_CHAT_URL)
    status = news_models.news_model_status([], ["2026-10-16"])
    assert [model["key"] for model in status["models"]] == ["event_guard", "grok", "classifier"]
    grok = status["models"][1]["detail"]
    assert grok["key_configured"] is False
    assert grok["endpoint"] == "xAI"
    assert status["models"][2]["detail"]["labeled_headlines"] > 0
