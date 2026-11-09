"""Jev shadow scores are recorded per pick and never raise into the scan."""

from types import SimpleNamespace

import requests
import pytest

from stock_picker.training import jev_shadow
from stock_picker.storage.jev_forecast_store import JevForecastStore


def signal(ticker="AAA", **extra):
    base = dict(ticker=ticker, predicted_return=0.012, open_price=10.5, prev_close=10.0,
                news_check={"judge": "grok", "flag": None, "articles": [{"headline": "Deal announced", "summary": "s"}]})
    return SimpleNamespace(**{**base, **extra})


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


ANSWERS = {
    "model": "jev-1.13.0",
    "answers": {name: {"type": "noul", "noul": value}
                for name, value in (("news_skip", 0.2), ("buy_today", 0.61), ("hold_overnight", 0.4))},
    "usage": {"input_tokens": 900, "output_tokens": 30},
}


def test_state_holds_only_open_known_facts():
    state = jev_shadow.build_state(signal(), "rank", 3, {"return_5d": 0.04, "rsi_14": float("nan")})
    assert state["list"] == "rank" and state["list_rank"] == 3
    assert abs(state["gap_from_prev_close"] - 0.05) < 1e-9
    assert state["recent"]["return_5d"] == 0.04
    assert state["recent"]["rsi_14"] is None
    assert state["headlines"] == ["Deal announced\nSummary: s"]
    assert "session_return" not in state and "close_price" not in state


def test_scores_every_pick_with_three_answers():
    session = FakeSession(FakeResponse(ANSWERS))
    scores = jev_shadow.shadow_scores(
        [(signal("AAA"), "rank", 1), (signal("BBB"), "fit", 1)],
        lambda ticker: {}, url="https://proxy.example/typesafe/v1/systemone", api_key="k", session=session,
    )
    assert set(scores) == {("rank", "AAA"), ("fit", "BBB")}
    assert scores[("rank", "AAA")]["buy_today"] == 0.61
    assert scores[("rank", "AAA")]["tokens"] == 930
    assert session.calls[0][1]["json"]["model"] == "jev-latest"
    assert set(session.calls[0][1]["json"]["questions"]) == {"news_skip", "buy_today", "hold_overnight"}


def test_failures_become_issues_not_exceptions():
    for response, issue in ((FakeResponse({}, status=503), "http_503"),
                            (requests.ConnectionError("https://proxy?key=secret"), "request_failed"),
                            (FakeResponse({"answers": {}}), "request_failed")):
        scores = jev_shadow.shadow_scores([(signal(), "fit", 1)], lambda ticker: {},
                                          url="u", api_key="k", session=FakeSession(response))
        assert scores[("fit", "AAA")]["issue"] == issue
        assert scores[("fit", "AAA")]["buy_today"] is None


def test_context_lookup_failure_still_asks_jev():
    def broken(ticker):
        raise FileNotFoundError(ticker)
    scores = jev_shadow.shadow_scores([(signal(), "fit", 1)], broken, url="u", api_key="k",
                                      session=FakeSession(FakeResponse(ANSWERS)))
    assert scores[("fit", "AAA")]["issue"] is None


def test_unconfigured_proxy_marks_every_pick():
    scores = jev_shadow.shadow_scores([(signal(), "fit", 1)], lambda ticker: {}, url="", api_key="k")
    assert scores[("fit", "AAA")]["issue"] == "not_configured"


def test_jev_url_uses_proxy_host_not_v1_base(monkeypatch):
    monkeypatch.setattr(jev_shadow, "llm_chat_url", lambda: "https://proxy.example/v1/chat/completions")
    assert jev_shadow.jev_url() == "https://proxy.example/typesafe/v1/systemone"
    monkeypatch.setattr(jev_shadow, "llm_chat_url", lambda: jev_shadow.XAI_CHAT_URL)
    assert jev_shadow.jev_url() is None


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), True, "0.6", None])
def test_invalid_probabilities_never_become_confidences(probability):
    body = {**ANSWERS, "answers": {**ANSWERS["answers"], "buy_today": {"type": "noul", "noul": probability}}}
    result = jev_shadow.ask_jev({}, url="u", api_key="k", session=FakeSession(FakeResponse(body)))
    assert result.issue == "request_failed"
    assert result.buy_today is None


def test_morning_forecast_archives_exact_request_before_scan_can_replace_it(tmp_path):
    store = JevForecastStore(tmp_path)
    session = FakeSession(FakeResponse(ANSWERS))
    scores = jev_shadow.shadow_scores([(signal(snapshot_date="2026-11-06"), "rank", 1)], lambda ticker: {},
                                      url="u", api_key="k", session=session, forecast_store=store)
    embedded = scores[("rank", "AAA")]
    record = store.read(embedded["forecast_id"])
    assert record["questions"] == session.calls[0][1]["json"]["questions"]
    assert record["state"] == session.calls[0][1]["json"]["state"]
    assert record["answers"]["buy_today"] == 0.61
    assert record["target"]["entry_basis"] == "session_open"
    assert record["target"]["entry_price"] == 10.5
    assert record["requested_at"] <= record["completed_at"]
    assert record["question_set_version"] == "morning_v1"


@pytest.mark.parametrize("metadata", [
    {"model": {"unexpected": "object"}}, {"model": []}, {"model": True}, {"model": ""}, {"model": "  "},
    {"usage": []}, {"usage": "930"}, {"usage": False},
    *({"usage": {"input_tokens": value, "output_tokens": 30}} for value in
      (True, "900", {}, -1, 1.5, 1.0, float("nan"), float("inf"), None)),
    {"usage": {"input_tokens": 900, "output_tokens": -1}},
])
def test_malformed_metadata_is_rejected_before_any_answer_is_returned(metadata):
    result = jev_shadow.ask_questions({}, jev_shadow.QUESTIONS, url="u", api_key="k",
                                     session=FakeSession(FakeResponse({**ANSWERS, **metadata})))
    assert result["issue"] == "request_failed"
    assert result["model"] is None and result["tokens"] is None
    assert all(value is None for value in result["answers"].values())


@pytest.mark.parametrize("metadata", [{}, {"model": None, "usage": None},
                                      {"usage": {}}, {"usage": {"input_tokens": 900}}])
def test_optional_metadata_stays_unknown_without_inventing_token_totals(metadata):
    result = jev_shadow.ask_questions({}, jev_shadow.QUESTIONS, url="u", api_key="k",
                                     session=FakeSession(FakeResponse({"answers": ANSWERS["answers"], **metadata})))
    assert result["issue"] is None
    assert result["model"] is None and result["tokens"] is None
    assert result["answers"]["buy_today"] == 0.61
