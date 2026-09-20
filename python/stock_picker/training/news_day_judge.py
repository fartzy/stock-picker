"""After Rank/Fit pick names: AVOID only on BIG news, good or bad.

Trial blow-up, FDA reject/approval, dilution, bankruptcy, takeover,
halt-pending-news -- skip the name. Analyst initiate, modest beat,
pre-market roundup -- trust the tape. Grok if keyed; else a Tfidf +
event-phrase logistic model. Not "any headline".
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import requests

from stock_picker.ingestion.finnhub_client import fetch_news_articles
from stock_picker.training.headline_sentiment import news_flag_from_articles as material_flag
from stock_picker.training.langfuse_local import trace_news_judge

XAI_API_KEY_ENV = "XAI_API_KEY"
XAI_KEY_FILE = Path.home() / ".config" / "api" / "xai.txt"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
LLM_PROXY_BASE_URL_ENV = "LLM_PROXY_BASE_URL"
LLM_PROXY_API_KEY_ENV = "LLM_PROXY_API_KEY"
GROK_CONFIG = Path.home() / ".grok" / "config.toml"
DEFAULT_PROXY_MODEL = "grok-4-fast-non-reasoning"
JUDGE_TIMEOUT_SECONDS = 20
MAX_HEADLINES = 5

_SYSTEM = (
    "We day-trade open-to-close using a price-pattern model. "
    "AVOID only if this is BIG company news that will dominate the session "
    "-- crash OR mania: trial hold/fail, FDA reject or approval, dilution/"
    "offering, bankruptcy, takeover/buyout, trading halt pending news, "
    "CEO ouster, guidance collapse. "
    "TRUST the tape for routine items: analyst initiate/upgrade, modest "
    "earnings beat, conference, index add, pre-market roundup that merely "
    "mentions the ticker. "
    "Reply with JSON only: {\"avoid\": true or false, \"reason\": \"short phrase\"}."
)


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
    env_name = str(cfg.get("env_key") or "").strip()
    if env_name:
        from_env = os.environ.get(env_name, "").strip()
        if from_env:
            return from_env
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
            out.append(headline)
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
    flagged = material_flag(articles)
    trace_news_judge(ticker, headlines, "tfidf", flagged is not None, flagged)
    return flagged


def fetch_recent_news_flags(tickers: list[str], as_of: date) -> dict[str, str]:
    """Picks only. ticker -> avoid reason. Empty dict if nothing to skip."""
    start = as_of - timedelta(days=1)
    while start.weekday() >= 5:
        start -= timedelta(days=1)
    articles_by_ticker = fetch_news_articles(tickers, as_of, from_date=start)
    flags: dict[str, str] = {}
    for ticker, articles in articles_by_ticker.items():
        flag = flag_from_articles(ticker, articles)
        if flag:
            flags[ticker] = flag
    return flags
