"""Skip inventory for morning news.

Gap-down + any material headline still skips. Some headlines skip even
when the open gapped up -- add a phrase here, not a one-off if in What if.
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
)


def always_skip_news(news_flag: str | None) -> bool:
    if not news_flag:
        return False
    text = news_flag.lower()
    return any(phrase in text for phrase in ALWAYS_SKIP_PHRASES)


def news_blocks_buy(
    news_flag: str | None,
    open_price: float | None,
    prev_close: float | None,
) -> bool:
    """Skip if the headline is on ALWAYS_SKIP_PHRASES, or news + gap down.

    Gap-up still buys for other material news. Missing prev_close does
    not block unless always_skip_news is true.
    """
    if not news_flag:
        return False
    if always_skip_news(news_flag):
        return True
    if open_price is None or prev_close is None or prev_close <= 0:
        return False
    return float(open_price) < float(prev_close)
