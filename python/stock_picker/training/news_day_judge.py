"""After Rank/Fit pick names: AVOID only on BIG news, good or bad.

Trial blow-up, sector/regulatory catalysts, FDA reject/approval, dilution, takeover,
halt-pending-news -- skip the name. Analyst initiate, modest beat,
pre-market roundup -- trust the tape. Grok if keyed; else a Tfidf +
event-phrase logistic model. Not "any headline".
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from stock_picker.ingestion.finnhub_client import MAX_NEWS_TICKERS, fetch_news_articles
from stock_picker.ingestion.news_sources import NewsCoverage, fetch_news_coverage
from stock_picker.news_check import NewsCheck
from stock_picker.news_skip import news_blocks_buy
from stock_picker.training.headline_sentiment import news_flag_from_articles as material_flag
from stock_picker.training.langfuse_local import trace_news_judge
from stock_picker.training.news_policy import NEWS_SYSTEM_PROMPT, regulatory_news_flag

XAI_API_KEY_ENV = "XAI_API_KEY"
XAI_KEY_FILE = Path.home() / ".config" / "api" / "xai.txt"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
LLM_PROXY_BASE_URL_ENV = "LLM_PROXY_BASE_URL"
LLM_PROXY_API_KEY_ENV = "LLM_PROXY_API_KEY"
GROK_CONFIG = Path.home() / ".grok" / "config.toml"
DEFAULT_PROXY_MODEL = "grok-4-fast-non-reasoning"
JUDGE_TIMEOUT_SECONDS = 20
MAX_HEADLINES = 20

_SYSTEM = NEWS_SYSTEM_PROMPT


def _grok_config_model() -> dict:
    """First matching block in ~/.grok/config.toml. Empty if missing/unreadable."""
    if not GROK_CONFIG.is_file():
        return {}
    try:
        import tomllib
    except ImportError:
        return {}
    try:
        raw = tomllib.loads(GROK_CONFIG.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    models = raw.get("model") or {}
    if not isinstance(models, dict):
        return {}
    preferred = os.environ.get("LLM_NEWS_MODEL", "").strip()
    if preferred in models and isinstance(models[preferred], dict):
        return models[preferred]
    for _name, block in models.items():
        if isinstance(block, dict) and block.get("base_url"):
            return block
    return {}


def llm_chat_url() -> str:
    """OpenAI-compatible chat completions URL. Proxy if LLM_PROXY_BASE_URL is set."""
    base = os.environ.get(LLM_PROXY_BASE_URL_ENV, "").strip().rstrip("/")
    if not base:
        cfg = _grok_config_model()
        base = str(cfg.get("base_url") or "").strip().rstrip("/")
    if base:
        return f"{base}/chat/completions"
    return XAI_CHAT_URL


def llm_model() -> str:
    override = os.environ.get("XAI_NEWS_MODEL") or os.environ.get("LLM_NEWS_MODEL")
    if override:
        return override
    cfg = _grok_config_model()
    if os.environ.get(LLM_PROXY_BASE_URL_ENV, "").strip() or cfg:
        return str(cfg.get("model") or DEFAULT_PROXY_MODEL)
    return "grok-4-fast-non-reasoning"


def llm_api_key(key_file: Path = XAI_KEY_FILE) -> str | None:
    proxy = os.environ.get(LLM_PROXY_API_KEY_ENV, "").strip()
    if proxy:
        return proxy
    cfg = _grok_config_model()
    configured_base = str(cfg.get("base_url") or "").strip().rstrip("/")
    override_base = os.environ.get(LLM_PROXY_BASE_URL_ENV, "").strip().rstrip("/")
    if override_base and override_base != configured_base:
        return xai_api_key(key_file=key_file) if llm_chat_url() == XAI_CHAT_URL else None
    # Grok supports an inline key in its owner-only local config. Read it in
    # memory; never copy credentials into scan payloads, logs, or the checkout.
    inline = str(cfg.get("api_key") or "").strip()
    if inline:
        return inline
    env_names = cfg.get("env_key") or []
    if isinstance(env_names, str):
        env_names = [env_names]
    for env_name in env_names:
        from_env = os.environ.get(env_name, "").strip()
        if from_env:
            return from_env
    # A personal xAI key must not be sent to a configured proxy by accident.
    if llm_chat_url() != XAI_CHAT_URL:
        return None
    return xai_api_key(key_file=key_file)


def xai_api_key(key_file: Path = XAI_KEY_FILE) -> str | None:
    env = os.environ.get(XAI_API_KEY_ENV, "").strip()
    if env:
        return env
    if not key_file.is_file():
        return None
    for line in key_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("XAI_API_KEY="):
            value = stripped.split("=", 1)[1].strip().strip("'\"")
            return value or None
        if ":" in stripped and stripped.split(":", 1)[0].count("@"):
            return stripped.split(":", 1)[1].strip() or None
        return stripped
    return None


def _headlines(articles: list[dict]) -> list[str]:
    out = []
    for article in articles[:MAX_HEADLINES]:
        headline = str(article.get("headline") or "").strip()
        if headline:
            summary = str(article.get("summary") or "").strip()
            out.append(f"{headline[:500]}\nSummary: {summary[:1500]}" if summary else headline[:500])
    return out


def _parse_judge(text: str) -> tuple[bool, str] | None:
    blob = (text or "").strip()
    match = re.search(r"\{.*\}", blob, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    avoid = payload.get("avoid")
    if avoid is True:
        reason = str(payload.get("reason") or "material news").strip()[:160]
        return True, reason
    if avoid is False:
        return False, ""
    return None


def grok_judge(ticker: str, headlines: list[str], api_key: str | None = None) -> tuple[bool, str] | None:
    """None = could not judge (no key, HTTP error, bad JSON)."""
    key = api_key if api_key is not None else llm_api_key()
    if not key or not headlines:
        return None
    numbered = "\n".join(f"- {h}" for h in headlines)
    user = (
        f"Ticker: {ticker}\n"
        f"This morning's headlines:\n{numbered}\n"
        "Is this BIG news we should not fade, or can we trust the price variation?"
    )
    try:
        response = requests.post(
            llm_chat_url(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": llm_model(),
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
            },
            timeout=JUDGE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return None
    return _parse_judge(content)


def flag_from_articles(ticker: str, articles: list[dict]) -> str | None:
    """BIG news only. Grok if keyed; else the material-news classifier."""
    headlines = _headlines(articles)
    if not headlines:
        return None
    judged = grok_judge(ticker, headlines)
    if judged is not None:
        avoid, reason = judged
        trace_news_judge(ticker, headlines, "grok", avoid, reason or None)
        return reason if avoid else None
    flagged = regulatory_news_flag(articles) or material_flag(articles)
    trace_news_judge(ticker, headlines, "tfidf", flagged is not None, flagged)
    return flagged


# Each name's judge is an independent LLM call (grok_judge -> requests.post,
# several seconds of network wait apiece); run them concurrently like the
# Finnhub article fetch above rather than serially -- on the morning path this
# loop was the single largest cost after row building, ~4-5s x every picked
# name. I/O-bound, so threads parallelize despite the GIL. Bounded to match
# fetch_news_articles' own fan-out, and pick counts are small (Rank top-20,
# a handful for Fit) so this is a wave or two, not a fixed 8-wide firehose.
NEWS_JUDGE_WORKERS = 8


def _judge_one(item: tuple[str, list[dict]]) -> tuple[str, str | None]:
    ticker, articles = item
    return ticker, flag_from_articles(ticker, articles)


def fetch_recent_news_flags(tickers: list[str], as_of: date) -> dict[str, str]:
    """Picks only. ticker -> avoid reason. Empty dict if nothing to skip."""
    start = as_of - timedelta(days=1)
    while start.weekday() >= 5:
        start -= timedelta(days=1)
    articles_by_ticker = fetch_news_articles(tickers, as_of, from_date=start)
    if not articles_by_ticker:
        return {}
    items = list(articles_by_ticker.items())
    workers = min(NEWS_JUDGE_WORKERS, len(items))
    flags: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ticker, flag in pool.map(_judge_one, items):
            if flag:
                flags[ticker] = flag
    return flags


def check_news_coverage(ticker: str, coverage: NewsCoverage) -> NewsCheck:
    """A completed attempt is not necessarily a successful news check."""
    articles = coverage.articles
    check = NewsCheck(
        article_count=len(articles), reviewed_count=0,
        sources=list(coverage.sources), issues=list(coverage.issues),
        checked_at=datetime.now(timezone.utc).isoformat(), articles=articles,
    )
    if not check.sources:
        check.status = "error"
        return check
    if not articles:
        check.status = "degraded" if check.issues else "no_news"
        return check
    headlines = _headlines(articles)
    judged = grok_judge(ticker, headlines)
    if judged is None:
        check.judge = "classifier"
        check.issues.append("llm_unavailable")
        check.flag = regulatory_news_flag(articles) or material_flag(articles)
        check.reviewed_count = len(articles)
    else:
        avoid, reason = judged
        check.judge = "grok"
        check.flag = reason if avoid else None
        check.reviewed_count = len(headlines)
        if len(articles) > len(headlines):
            check.issues.append("review_truncated")
            unseen = articles[MAX_HEADLINES:]
            check.flag = check.flag or regulatory_news_flag(unseen) or material_flag(unseen)
    check.status = "degraded" if check.issues else "complete"
    trace_news_judge(ticker, headlines, check.judge, bool(check.flag), check.flag)
    return check


def fetch_recent_news_checks(tickers: list[str], as_of: date) -> dict[str, NewsCheck]:
    """Prior weekday through now/as_of, using both feeds for every pick."""
    names = list(dict.fromkeys(tickers))
    if not names:
        return {}
    start = as_of - timedelta(days=1)
    while start.weekday() >= 5:
        start -= timedelta(days=1)
    selected = names[:MAX_NEWS_TICKERS]
    coverage = fetch_news_coverage(selected, start, as_of)

    def check_one(ticker: str) -> tuple[str, NewsCheck]:
        evidence = coverage.get(ticker, NewsCoverage(issues=["feeds_unavailable"]))
        return ticker, check_news_coverage(ticker, evidence)

    with ThreadPoolExecutor(max_workers=min(NEWS_JUDGE_WORKERS, len(selected))) as pool:
        results = dict(pool.map(check_one, selected))
    results.update({
        name: NewsCheck(status="not_checked", issues=["ticker_limit"])
        for name in names[MAX_NEWS_TICKERS:]
    })
    return results
