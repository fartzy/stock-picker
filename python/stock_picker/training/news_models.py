"""Status of the news models that gate picks, for the Models tab.

Three deciders run in order on each picked ticker (`check_news_coverage`):
the event-phrase guard, the Grok LLM judge, then the TF-IDF + logistic
headline classifier as fallback. None of them has a trained artifact or a
measured accuracy, so the evidence here is how often each one decided and
what it flagged across saved morning scans.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from stock_picker.training import headline_sentiment
from stock_picker.training.news_day_judge import (
    JUDGE_TIMEOUT_SECONDS,
    MAX_HEADLINES,
    XAI_CHAT_URL,
    llm_api_key,
    llm_chat_url,
    llm_model,
)

# Judge labels written to NewsCheck.judge by check_news_coverage.
EVENT_GUARD = "event_guard"
GROK = "grok"
CLASSIFIER = "classifier"
RECENT_SCAN_DAYS = 20


def _checks(payloads: Iterable[dict]) -> list[dict]:
    return [
        signal["news_check"]
        for payload in payloads
        for signal in payload.get("signals") or []
        if isinstance(signal.get("news_check"), dict)
    ]


def summarize_news_checks(payloads: Iterable[dict]) -> dict:
    """Counts by decider and status across saved scan payloads."""
    checks = _checks(payloads)
    judged = Counter(check.get("judge") or "none" for check in checks)
    flagged = Counter(check.get("judge") or "none" for check in checks if check.get("flag"))
    return {
        "checked": len(checks),
        "flagged": sum(flagged.values()),
        "by_judge": {name: {"decided": judged[name], "flagged": flagged[name]} for name in judged},
        "by_status": dict(Counter(check.get("status") or "unknown" for check in checks)),
        "llm_unavailable": sum("llm_unavailable" in (check.get("issues") or []) for check in checks),
    }


def news_model_status(payloads: Iterable[dict], scan_days: list[str]) -> dict:
    """Config for each news decider plus decision counts from recent scans."""
    url = llm_chat_url()
    return {
        "scan_days": scan_days,
        "usage": summarize_news_checks(payloads),
        "models": [
            {
                "key": EVENT_GUARD,
                "name": "Event guard",
                "kind": "Phrase rules",
                "role": "Runs first. Flags corporate actions, regulatory events and clinical readouts.",
                "source": "python/stock_picker/training/news_policy.py",
                "detail": {},
            },
            {
                "key": GROK,
                "name": "Grok news judge",
                "kind": "LLM",
                "role": f"AVOID or OK from up to {MAX_HEADLINES} headlines when a key is configured.",
                "source": "python/stock_picker/training/news_day_judge.py",
                "detail": {
                    "model": llm_model(),
                    "endpoint": "xAI" if url == XAI_CHAT_URL else "Proxy",
                    "key_configured": llm_api_key() is not None,
                    "timeout_seconds": JUDGE_TIMEOUT_SECONDS,
                },
            },
            {
                "key": CLASSIFIER,
                "name": "Headline classifier",
                "kind": "TF-IDF + logistic regression",
                "role": "Fallback when Grok is unavailable. Also scores every ingested article.",
                "source": "python/stock_picker/training/headline_sentiment.py",
                "detail": {
                    "labeled_headlines": len(headline_sentiment._LABELED),
                    "event_phrases": len(headline_sentiment._MATERIAL_PHRASES),
                },
            },
        ],
    }
