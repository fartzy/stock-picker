import threading
from datetime import date

from stock_picker.training.news_day_judge import (
    _parse_judge,
    fetch_recent_news_flags,
    flag_from_articles,
    grok_judge,
    llm_chat_url,
    llm_model,
    news_blocks_buy,
)


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
