"""After Rank/Fit pick names, ask a small LLM: news day or trust the tape?

Finnhub fetches headlines for those tickers only. Grok (if XAI_API_KEY or
~/.config/api/xai.txt) answers a yes/no. No key / timeout -> sklearn
classifier. Test run does not call this.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import requests

from stock_picker.ingestion.finnhub_client import fetch_news_articles
from stock_picker.training.headline_sentiment import news_flag_from_articles as sklearn_flag

XAI_API_KEY_ENV = "XAI_API_KEY"
XAI_KEY_FILE = Path.home() / ".config" / "api" / "xai.txt"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = os.environ.get("XAI_NEWS_MODEL", "grok-4-fast-non-reasoning")
JUDGE_TIMEOUT_SECONDS = 20
MAX_HEADLINES = 5

_SYSTEM = (
    "We day-trade open-to-close using a price-pattern model. "
    "Your only job: for this ticker, is this morning a NEWS day "
    "(trial hold, FDA, offering, bankruptcy, unexpected company event) "
    "so we should AVOID and not trust the model's price variation, "
    "or is the news routine (earnings beat, analyst note, CEO thanking staff) "
    "so we can TRUST the tape? "
    "Reply with JSON only: {\"avoid\": true or false, \"reason\": \"short phrase\"}."
)


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
    key = api_key if api_key is not None else xai_api_key()
    if not key or not headlines:
        return None
    numbered = "\n".join(f"- {h}" for h in headlines)
    user = (
        f"Ticker: {ticker}\n"
        f"This morning's headlines:\n{numbered}\n"
        "Is there too much news this morning, or can we trust the price variation?"
    )
    try:
        response = requests.post(
            XAI_CHAT_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": XAI_MODEL,
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
    headlines = _headlines(articles)
    judged = grok_judge(ticker, headlines)
    if judged is not None:
        avoid, reason = judged
        return reason if avoid else None
    return sklearn_flag(articles)


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
