"""Shared morning news decision for live picks and historical What if.

Incomplete reviews are holds. Hard-stop events skip regardless of gap; other
material headlines skip when the open gaps down.
"""

from __future__ import annotations

# Skip even on a gap-up. First item: insider / Form 4 selling.
ALWAYS_SKIP_PHRASES: tuple[str, ...] = (
    "insider sell",
    "insider sold",
    "insider sale",
    "form 4",
    "cto sells",
    "cto sold",
    "cfo sells",
    "cfo sold",
    "ceo sells",
    "ceo sold",
    "director sells",
    "director sold",
    "officer sells",
    "officer sold",
    "sold shares",
    "sells shares",
    "share sale",
    "stock sale",
    "corporate action:",
    "clinical readout:",
)

INCOMPLETE_NEWS_STATUSES = frozenset({"degraded", "error", "not_checked", "unknown"})


def always_skip_news(news_flag: str | None) -> bool:
    if not news_flag:
        return False
    text = news_flag.lower()
    return any(phrase in text for phrase in ALWAYS_SKIP_PHRASES)


def news_blocks_buy(
    news_flag: str | None,
    open_price: float | None,
    prev_close: float | None,
    check_status: str | None = None,
) -> bool:
    """Hold incomplete reviews; skip hard-stop events or material gap-downs.

    Gap-up still buys for other material news after a complete review.
    Missing prev_close does not block unless the event or review status does.
    """
    if check_status in INCOMPLETE_NEWS_STATUSES:
        return True
    if not news_flag:
        return False
    if always_skip_news(news_flag):
        return True
    if open_price is None or prev_close is None or prev_close <= 0:
        return False
    return float(open_price) < float(prev_close)
