"""Optional traces to a *local* Langfuse. Never cloud.langfuse.com.

If LANGFUSE_BASE_URL is unset we default to http://127.0.0.1:3100.
No keys / Langfuse down / timeout -> silent no-op. Morning scoring
must not fail because tracing is off.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:3100"
DEFAULT_PUBLIC_KEY = "pk-lf-local-stockpicker"
DEFAULT_SECRET_KEY = "sk-lf-local-stockpicker"


def langfuse_base_url() -> str:
    return os.environ.get("LANGFUSE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _auth() -> tuple[str, str]:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", DEFAULT_PUBLIC_KEY)
    secret = os.environ.get("LANGFUSE_SECRET_KEY", DEFAULT_SECRET_KEY)
    return public, secret


def trace_news_judge(
    ticker: str,
    headlines: list[str],
    judge: str,
    avoid: bool | None,
    reason: str | None,
) -> None:
    """Best-effort ingest. Swallow every failure."""
    if os.environ.get("LANGFUSE_DISABLED", "").strip() in {"1", "true", "yes"}:
        return
    base = langfuse_base_url()
    if not base.startswith("http://127.0.0.1") and not base.startswith("http://localhost"):
        return
    now = datetime.now(timezone.utc).isoformat()
    body = {
        "batch": [
            {
                "id": f"news-{ticker}-{now}",
                "timestamp": now,
                "type": "trace-create",
                "body": {
                    "id": f"news-{ticker}-{now}",
                    "name": "news_day_judge",
                    "timestamp": now,
                    "input": {"ticker": ticker, "headlines": headlines, "judge": judge},
                    "output": {"avoid": avoid, "reason": reason},
                    "metadata": {"local_only": True},
                },
            }
        ]
    }
    public, secret = _auth()
    try:
        requests.post(
            f"{base}/api/public/ingestion",
            json=body,
            auth=(public, secret),
            timeout=1.5,
        )
    except requests.RequestException:
        return
