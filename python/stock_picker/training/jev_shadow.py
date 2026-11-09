"""Shadow Jev confidences on each morning pick. Recorded, never acted on.

TypeSafe Jev answers typed questions with probabilities to validate locally. For each
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
from uuid import uuid4

import requests

from stock_picker.ingestion.session import cash_session_window, US_EASTERN
from stock_picker.training.overnight_dataset import next_expected_session
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
    forecast_id: str | None = None
    requested_at: str | None = None
    completed_at: str | None = None
    question_set_version: str = "morning_v1"


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
    """Original open-to-close question with evidence known when the request runs."""
    news = signal.news_check or {}
    gap = None
    if signal.prev_close and signal.open_price:
        gap = signal.open_price / signal.prev_close - 1
    return {
        "ticker": signal.ticker,
        "list": kind,
        "list_rank": rank,
        "model_score": _finite(signal.predicted_return),
        "model_score_semantics": "relative_ranking_score" if kind == "rank" else "predicted_open_to_close_return",
        "open_price": _finite(signal.open_price),
        "prev_close": _finite(signal.prev_close),
        "gap_from_prev_close": _finite(gap),
        "recent": {name: _finite(context.get(name)) for name in CONTEXT_COLUMNS},
        "headlines": _headlines(news.get("articles") or []),
        "news_judge": {"judge": news.get("judge"), "flag": news.get("flag")},
        "news_evidence": news,
        "feature_snapshot_date": getattr(signal, "snapshot_date", None),
    }


def ask_questions(state: dict, questions: dict, *, url: str | None, api_key: str | None,
                  session=requests, now_fn=None) -> dict:
    """SystemOne wire contract shared by versioned forecasts, without policy gates.

    Model/usage metadata is optional. Missing model or incomplete token usage
    remains unknown; supplied values must satisfy the wire types before any
    answer can enter the immutable journal.
    """
    now = now_fn or (lambda: datetime.now(timezone.utc))
    result = {"answers": {name: None for name in questions}, "model": None,
              "tokens": None, "issue": None, "requested_at": now().isoformat()}
    if not url or not api_key:
        result.update(issue="not_configured", completed_at=now().isoformat())
        return result
    try:
        response = session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": JEV_MODEL, "state": state, "questions": questions},
            timeout=JEV_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        answers = body["answers"]
        model = body.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("invalid model metadata")
        usage = body.get("usage")
        tokens = None
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("invalid usage metadata")
            for name in ("input_tokens", "output_tokens"):
                if name in usage and (type(usage[name]) is not int or usage[name] < 0):
                    raise ValueError("invalid token count")
            if "input_tokens" in usage and "output_tokens" in usage:
                tokens = usage["input_tokens"] + usage["output_tokens"]
        probabilities = {}
        for name in questions:
            answer = answers[name]
            value = answer["noul"]
            if answer.get("type") != "noul" or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid probability")
            probabilities[name] = float(value)
        result.update(answers=probabilities, model=model, tokens=tokens)
    except requests.HTTPError as exc:
        result["issue"] = f"http_{exc.response.status_code if exc.response is not None else 'error'}"
    except (requests.RequestException, KeyError, TypeError, ValueError, AttributeError, OverflowError):
        # Raw errors can echo request URLs or headers; keep a stable code only.
        result["issue"] = "request_failed"
    result["completed_at"] = now().isoformat()
    return result


def ask_jev(state: dict, *, url: str, api_key: str, session=requests) -> JevShadow:
    result = ask_questions(state, QUESTIONS, url=url, api_key=api_key, session=session)
    return JevShadow(**result["answers"], model=result["model"], tokens=result["tokens"],
                     issue=result["issue"], asked_at=result["completed_at"],
                     requested_at=result["requested_at"], completed_at=result["completed_at"])


def shadow_scores(
    picks: list[tuple[object, str, int]],
    context_for,
    *,
    url: str | None = None,
    api_key: str | None = None,
    session=requests,
    forecast_store=None,
) -> dict[tuple[str, str], dict]:
    """{(kind, ticker): JevShadow dict} for (signal, kind, rank) picks.

    `context_for(ticker)` returns the latest feature values; failures there or
    from Jev become an `issue` on that pick, never an exception.
    """
    url = url if url is not None else jev_url()
    api_key = api_key if api_key is not None else llm_api_key()
    def one(pick):
        signal, kind, rank = pick
        try:
            context = context_for(signal.ticker) or {}
        except Exception:
            context = {}
        state = build_state(signal, kind, rank, context)
        evidence_at = datetime.now(timezone.utc)
        state["evidence_as_of"] = evidence_at.isoformat()
        result = ask_jev(state, url=url, api_key=api_key, session=session)
        if forecast_store is not None:
            window = cash_session_window(evidence_at)
            day = evidence_at.astimezone(US_EASTERN).date()
            following = next_expected_session(day)
            news = signal.news_check or {}
            issues = []
            if window is not None and evidence_at >= window.open_at:
                issues.append("forecast_after_entry")
            if window is None or datetime.fromisoformat(result.completed_at) >= window.close_at:
                result.issue = result.issue or "session_ended_before_answer"
            result.forecast_id = str(uuid4())
            forecast_store.append({
                "id": result.forecast_id, "schema_version": 1, "question_set_version": "morning_v1",
                "ticker": signal.ticker, "session": day.isoformat(), "kind": kind,
                "list_rank": rank, "score": _finite(signal.predicted_return),
                "requested_at": result.requested_at, "completed_at": result.completed_at,
                "evidence_as_of": evidence_at.isoformat(), "state": state, "questions": QUESTIONS,
                "answers": {name: getattr(result, name) for name in QUESTIONS},
                "model": result.model, "tokens": result.tokens, "issue": result.issue,
                "target": {"entry_price": _finite(signal.open_price), "entry_basis": "session_open",
                           "entry_observed_at": window.open_at.isoformat() if window else None,
                           "exit_at": window.close_at.isoformat() if window else None,
                           "next_session": following.isoformat() if following else None,
                           "round_trip_cost_per_share": 0.0},
                "coverage": {"news_status": news.get("status", "unknown"),
                             "news_checked_at": news.get("checked_at"),
                             "news_sources": news.get("sources", []), "news_issues": news.get("issues", []),
                             "feature_snapshot_date": getattr(signal, "snapshot_date", None), "issues": issues},
            })
        return (kind, signal.ticker), asdict(result)

    with ThreadPoolExecutor(max_workers=JEV_WORKERS) as pool:
        return dict(pool.map(one, picks))
