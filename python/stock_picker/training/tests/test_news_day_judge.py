import threading
from datetime import date
from types import SimpleNamespace

from stock_picker.ingestion.news_sources import NewsCoverage
from stock_picker.training.news_policy import high_impact_event_flag, regulatory_news_flag
from stock_picker.news_check import NewsCheck, apply_news_checks

from stock_picker.training.news_day_judge import (
    _parse_judge,
    check_news_coverage,
    fetch_recent_news_flags,
    fetch_recent_news_checks,
    flag_from_articles,
    grok_judge,
    llm_chat_url,
    llm_api_key,
    llm_model,
    news_blocks_buy,
)


def test_local_grok_key_works_without_shell_environment(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[model.news]\nbase_url = "https://proxy.example/v1"\napi_key = "local-test-key"\n')
    monkeypatch.setattr("stock_picker.training.news_day_judge.GROK_CONFIG", config)
    monkeypatch.delenv("LLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROXY_BASE_URL", raising=False)
    assert llm_api_key(key_file=tmp_path / "missing") == "local-test-key"
    assert llm_chat_url() == "https://proxy.example/v1/chat/completions"
    assert list(tmp_path.iterdir()) == [config]


def test_grok_config_supports_ordered_environment_names(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[model.news]\nbase_url = "https://proxy.example/v1"\nenv_key = ["NEWS_TEST_EMPTY", "NEWS_TEST_KEY"]\n')
    monkeypatch.setattr("stock_picker.training.news_day_judge.GROK_CONFIG", config)
    monkeypatch.delenv("LLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_TEST_EMPTY", raising=False)
    monkeypatch.setenv("NEWS_TEST_KEY", "environment-test-key")
    assert llm_api_key(key_file=tmp_path / "missing") == "environment-test-key"


def test_personal_xai_key_is_not_used_for_a_proxy(tmp_path, monkeypatch):
    monkeypatch.setattr("stock_picker.training.news_day_judge.GROK_CONFIG", tmp_path / "missing")
    monkeypatch.delenv("LLM_PROXY_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROXY_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("XAI_API_KEY", "personal-test-key")
    assert llm_api_key() is None


def test_config_key_stays_bound_to_its_own_proxy(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[model.news]\nbase_url = "https://proxy.example/v1"\napi_key = "local-test-key"\n')
    monkeypatch.setattr("stock_picker.training.news_day_judge.GROK_CONFIG", config)
    monkeypatch.delenv("LLM_PROXY_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROXY_BASE_URL", "https://different.example/v1")
    assert llm_api_key() is None


def test_llm_chat_url_uses_proxy_env(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_BASE_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("LLM_NEWS_MODEL", "local-test-model")
    assert llm_chat_url() == "http://127.0.0.1:4000/v1/chat/completions"
    assert llm_model() == "local-test-model"


def test_news_blocks_buy_only_when_open_gaps_down():
    assert news_blocks_buy("PIPE", 9.0, 10.0) is True
    assert news_blocks_buy("PIPE", 11.0, 10.0) is False
    assert news_blocks_buy("PIPE", 10.0, 10.0) is False
    assert news_blocks_buy(None, 9.0, 10.0) is False
    assert news_blocks_buy("PIPE", 9.0, None) is False


def test_insider_sell_skips_even_on_gap_up():
    headline = (
        "Fastly's CTO Sells Over 33,000 Shares for $825,000 "
        "After the Stock Rose 177% Over the Past Year"
    )
    assert news_blocks_buy(headline, 28.02, 27.42) is True
    assert news_blocks_buy("PIPE dilution/offering", 11.0, 10.0) is False


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


def test_fetch_recent_news_flags_judges_all_and_keeps_only_flagged(monkeypatch):
    # The judge loop runs concurrently across tickers (ThreadPoolExecutor);
    # confirm every judged name is aggregated and only the flagged ones survive.
    articles = {t: [{"headline": f"{t} headline"}] for t in ["AAA", "BBB", "CCC", "DDD"]}
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.fetch_news_articles",
        lambda *a, **k: articles,
    )
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.flag_from_articles",
        lambda ticker, arts: "avoid" if ticker in {"BBB", "DDD"} else None,
    )
    flags = fetch_recent_news_flags(["AAA", "BBB", "CCC", "DDD"], date(2026, 9, 25))
    assert flags == {"BBB": "avoid", "DDD": "avoid"}


def test_fetch_recent_news_flags_judges_names_concurrently(monkeypatch):
    """Serial judging would deadlock: each call waits for the others to start."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    barrier = threading.Barrier(len(tickers), timeout=2)
    seen: list[str] = []
    lock = threading.Lock()

    def flag(ticker, _arts):
        with lock:
            seen.append(ticker)
        barrier.wait()
        return "avoid" if ticker == "BBB" else None

    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.fetch_news_articles",
        lambda *a, **k: {ticker: [{"headline": ticker}] for ticker in tickers},
    )
    monkeypatch.setattr("stock_picker.training.news_day_judge.flag_from_articles", flag)

    flags = fetch_recent_news_flags(tickers, date(2026, 9, 25))

    assert flags == {"BBB": "avoid"}
    assert set(seen) == set(tickers)


DEA_HEADLINE = "MSOS, TRLV Slide After Hours As DEA Judge Pauses Cannabis Rescheduling Case"


def test_regulatory_catalyst_is_flagged_even_without_llm(monkeypatch):
    monkeypatch.setattr("stock_picker.training.news_day_judge.grok_judge", lambda *a: None)
    result = check_news_coverage("TRLV", NewsCoverage(
        articles=[{"headline": DEA_HEADLINE}], sources=["finnhub", "polygon"],
    ))
    assert result.flag == DEA_HEADLINE
    assert result.status == "degraded"
    assert result.article_count == 1
    assert "llm_unavailable" in result.issues
    assert news_blocks_buy(result.flag, 10.85, 12.41)
    assert not news_blocks_buy(result.flag, 13, 12.41)


def test_spin_off_is_a_hard_stop_even_when_the_headline_is_generic(monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.news_day_judge.grok_judge",
        lambda *a: (_ for _ in ()).throw(AssertionError("hard stops should not depend on AI")),
    )
    result = check_news_coverage("CTVA", NewsCoverage(
        articles=[{
            "headline": "Dear Corteva Stock Fans, Mark Your Calendars for Oct. 1",
            "summary": "Corteva is about to officially spin off its seed genetics business",
        }],
        sources=["finnhub", "polygon"],
    ))
    assert result.flag.startswith("corporate action:")
    assert result.judge == "event_guard"
    assert result.status == "complete"
    assert news_blocks_buy(result.flag, 14.44, 77.65, result.status)
    assert news_blocks_buy(result.flag, 80.0, 77.65, result.status)


def test_released_phase_two_data_is_a_hard_stop_regardless_of_tone(monkeypatch):
    monkeypatch.setattr("stock_picker.training.news_day_judge.grok_judge", lambda *a: (False, ""))
    result = check_news_coverage("NKTR", NewsCoverage(
        articles=[{
            "headline": "New Biomarker Data Presented at EADV Congress",
            "summary": "Nektar announced new results from its Phase 2b study in patients.",
        }],
        sources=["finnhub"],
    ))
    assert result.flag.startswith("clinical readout:")
    assert result.judge == "event_guard"
    assert news_blocks_buy(result.flag, 60.0, 55.0, result.status)


def test_future_conference_and_generic_spin_off_discussion_are_not_hard_stops():
    assert high_impact_event_flag([{
        "headline": "Company will present Phase 2b trial data at a conference next month",
    }]) is None
    assert high_impact_event_flag([{
        "headline": "Analysts discuss whether the company could spin off a unit someday",
    }]) is None


def test_regulatory_fallback_distinguishes_actions_from_speculation():
    for headline in [
        "Court blocks cannabis legalization case", "Regulator revokes operating license",
        "Government bans chip exports", "Government approves cannabis rescheduling",
    ]:
        assert regulatory_news_flag([{"headline": headline}]) == headline
    for headline in [
        "DEA judge may pause cannabis rescheduling", "Court does not block cannabis legalization",
        "Cannabis rescheduling debate continues", "Analyst upgrades cannabis stocks",
        "Government considers whether regulator revokes operating license",
    ]:
        assert regulatory_news_flag([{"headline": headline}]) is None


def test_sixth_article_and_summary_reach_judge(monkeypatch):
    seen = []
    monkeypatch.setattr("stock_picker.training.news_day_judge.grok_judge", lambda ticker, text: seen.extend(text) or (True, "Regulatory pause"))
    articles = [{"headline": f"Roundup {i}"} for i in range(5)] + [{"headline": DEA_HEADLINE, "summary": "A sector-wide regulatory pause."}]
    result = check_news_coverage("TRLV", NewsCoverage(articles=articles, sources=["finnhub", "polygon"]))
    assert len(seen) == 6
    assert "sector-wide regulatory pause" in seen[-1]
    assert result.status == "complete"
    assert result.reviewed_count == 6


def test_feed_failures_and_empty_success_cannot_claim_complete(monkeypatch):
    monkeypatch.setattr("stock_picker.training.news_day_judge.grok_judge", lambda *a: (False, ""))
    assert check_news_coverage("X", NewsCoverage(issues=["feeds_unavailable"])).status == "error"
    assert check_news_coverage("X", NewsCoverage(sources=["finnhub", "polygon"])).status == "no_news"
    result = check_news_coverage("X", NewsCoverage(articles=[{"headline": "Routine"}], sources=["finnhub"], issues=["polygon_unavailable"]))
    assert result.status == "degraded"
    assert result.flag is None


def test_truncated_review_stays_degraded_even_if_llm_says_no_flag(monkeypatch):
    monkeypatch.setattr("stock_picker.training.news_day_judge.grok_judge", lambda *a: (False, ""))
    articles = [{"headline": "Routine quarterly update"} for _ in range(20)] + [{"headline": DEA_HEADLINE}]
    result = check_news_coverage("TRLV", NewsCoverage(articles=articles, sources=["finnhub", "polygon"]))
    assert result.status == "degraded"
    assert result.flag == DEA_HEADLINE
    assert result.reviewed_count == 20
    assert result.article_count == 21


def test_grok_http_failure_does_not_escape_or_expose_request(monkeypatch):
    import requests
    def fail(*a, **kw):
        raise requests.HTTPError("provider rejected request")
    monkeypatch.setattr("stock_picker.training.news_day_judge.requests.post", fail)
    assert grok_judge("TRLV", [DEA_HEADLINE], api_key="dummy-test-key") is None


def test_excess_names_are_explicitly_not_checked(monkeypatch):
    fetched = []
    def fetch(names, start, end):
        fetched.extend(names)
        return {name: NewsCoverage(sources=["finnhub", "polygon"]) for name in names}
    monkeypatch.setattr("stock_picker.training.news_day_judge.fetch_news_coverage", fetch)
    results = fetch_recent_news_checks([f"T{i}" for i in range(41)], date(2026, 9, 30))
    assert len(fetched) == 40
    assert results["T40"].status == "not_checked"
    assert results["T40"].issues == ["ticker_limit"]


def test_failed_check_carries_evidence_to_each_list_without_claiming_checked():
    signals = [SimpleNamespace(ticker="TRLV"), SimpleNamespace(ticker="TRLV")]
    apply_news_checks(signals, ["TRLV"], {"TRLV": NewsCheck(status="error", issues=["check_failed"])})
    assert all(not signal.news_checked for signal in signals)
    assert all(signal.news_check["status"] == "error" for signal in signals)
    assert all(signal.news_flag is None for signal in signals)
