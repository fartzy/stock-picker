"""Shadow Jev confidences on each morning pick. Recorded, never acted on.

TypeSafe Jev answers typed questions with calibrated probabilities. For each
Rank/Fit pick we ask three yes/no questions in one call -- skip for news, buy
at today's open, hold past the close to tomorrow's open -- and save the
answers beside the pick. Nothing reads them back into picks, skips or sizing
until outcomes show they add something; see ADR 0025.

Same LLM proxy and key as the Grok news judge. Jev sits on the proxy host's
`/typesafe/v1/systemone` route, not under the OpenAI-style `/v1` base.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

from stock_picker.training.news_day_judge import XAI_CHAT_URL, _headlines, llm_api_key, llm_chat_url
from stock_picker.training.news_policy import NEWS_SYSTEM_PROMPT

JEV_MODEL = "jev-latest"
JEV_PATH = "/typesafe/v1/systemone"
JEV_TIMEOUT_SECONDS = 15
JEV_WORKERS = 8

# Per-ticker market context sent as Jev's state. Names match feature columns.
CONTEXT_COLUMNS = (
    "return_1d", "return_5d", "return_20d", "rsi_14", "atr_14",
    "price_vs_sma_20d", "price_vs_sma_50d", "volume_ratio_20d",
)

QUESTIONS = {
    "news_skip": {
        "type": "noul",
        "instructions": NEWS_SYSTEM_PROMPT,
        "criteria": {
            "true": "AVOID: material company or sector news directly affecting this ticker today",
            "false": "OK: routine coverage, roundups, analyst notes, or nothing material",
        },
    },
    "buy_today": {
        "type": "noul",
        "instructions": (
            "We buy at today's open and sell at today's close. Given the model scores, "
            "the gap from yesterday's close, recent price behaviour and the headlines, "
            "will this stock close above its open today?"
        ),
        "criteria": {"true": "Closes above today's open", "false": "Closes at or below today's open"},
    },
    "hold_overnight": {
        "type": "noul",
        "instructions": (
            "Instead of selling at today's close we could hold one night and sell at "
            "tomorrow's open. Given the same evidence, will tomorrow's open be above "
            "today's close?"
        ),
        "criteria": {"true": "Tomorrow opens above today's close", "false": "Tomorrow opens at or below today's close"},
    },
}


@dataclass
class JevShadow:
    news_skip: float | None = None
    buy_today: float | None = None
    hold_overnight: float | None = None
    model: str | None = None
    tokens: int | None = None
    issue: str | None = None
    asked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def jev_url() -> str | None:
    """Jev route on the configured LLM proxy host. None when only direct xAI is set."""
    chat = llm_chat_url()
    if chat == XAI_CHAT_URL:
        return None
    parts = urlsplit(chat)
    return f"{parts.scheme}://{parts.netloc}{JEV_PATH}"


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def build_state(signal, kind: str, rank: int, context: dict) -> dict:
    """Everything Jev sees for one pick. Only facts known at the open."""
    news = signal.news_check or {}
    gap = None
    if signal.prev_close and signal.open_price:
        gap = signal.open_price / signal.prev_close - 1
    return {
        "ticker": signal.ticker,
        "list": kind,
        "list_rank": rank,
        "model_score": _finite(signal.predicted_return),
        "open_price": _finite(signal.open_price),
        "prev_close": _finite(signal.prev_close),
        "gap_from_prev_close": _finite(gap),
        "recent": {name: _finite(context.get(name)) for name in CONTEXT_COLUMNS},
        "headlines": _headlines(news.get("articles") or []),
        "news_judge": {"judge": news.get("judge"), "flag": news.get("flag")},
    }


def ask_jev(state: dict, *, url: str, api_key: str, session=requests) -> JevShadow:
    try:
        response = session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": JEV_MODEL, "state": state, "questions": QUESTIONS},
            timeout=JEV_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        answers = body["answers"]
        usage = body.get("usage") or {}
        return JevShadow(
            **{name: _finite(answers[name]["noul"]) for name in QUESTIONS},
            model=body.get("model"),
            tokens=(usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        )
    except requests.HTTPError as exc:
        return JevShadow(issue=f"http_{exc.response.status_code if exc.response is not None else 'error'}")
    except (requests.RequestException, KeyError, TypeError, ValueError):
        # Raw errors can echo request URLs or headers; keep a stable code only.
        return JevShadow(issue="request_failed")


def shadow_scores(
    picks: list[tuple[object, str, int]],
    context_for,
    *,
    url: str | None = None,
    api_key: str | None = None,
    session=requests,
) -> dict[tuple[str, str], dict]:
    """{(kind, ticker): JevShadow dict} for (signal, kind, rank) picks.

    `context_for(ticker)` returns the latest feature values; failures there or
    from Jev become an `issue` on that pick, never an exception.
    """
    url = url if url is not None else jev_url()
    api_key = api_key if api_key is not None else llm_api_key()
    if not url or not api_key:
        return {(kind, signal.ticker): asdict(JevShadow(issue="not_configured")) for signal, kind, _ in picks}

    def one(pick):
        signal, kind, rank = pick
        try:
            context = context_for(signal.ticker) or {}
        except Exception:
            context = {}
        result = ask_jev(build_state(signal, kind, rank, context), url=url, api_key=api_key, session=session)
        return (kind, signal.ticker), asdict(result)

    with ThreadPoolExecutor(max_workers=JEV_WORKERS) as pool:
        return dict(pool.map(one, picks))
