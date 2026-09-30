"""News-check evidence shared by live scans, the API, and the paper book."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

NewsStatus = Literal["complete", "no_news", "degraded", "error", "not_checked", "unknown"]


@dataclass
class NewsCheck:
    status: NewsStatus = "unknown"
    flag: str | None = None
    article_count: int | None = None
    reviewed_count: int | None = None
    judge: str | None = None
    sources: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    checked_at: str | None = None
    # Exact normalized inputs, for explaining a past decision without relying
    # on a feed that may have changed since the morning scan.
    articles: list[dict] = field(default_factory=list)


def news_check_payload(result: NewsCheck | str | None) -> dict:
    if isinstance(result, NewsCheck):
        return asdict(result)
    # Legacy injected flag-only fetchers provide no proof of feed coverage.
    return asdict(NewsCheck(flag=result if isinstance(result, str) else None))


def apply_news_checks(signals, names: list[str], results: dict) -> None:
    wanted = set(names)
    for signal in signals:
        if signal.ticker not in wanted:
            continue
        payload = news_check_payload(results.get(signal.ticker))
        signal.news_flag = payload["flag"]
        signal.news_check = payload
        signal.news_checked = payload["status"] in {"complete", "no_news"}
