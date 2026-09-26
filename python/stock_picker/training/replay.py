"""Replay a pickle on a past morning's official opens.

Uses PriceStore daily bars (not Yahoo v7). Features are the last snapshot
strictly before `as_of`. Fit is the archived run if `model_run_id` is set.
Rank is still the latest rank pickle (ListFold) -- we do not archive Rank per run.
"""

from __future__ import annotations

import uuid
from datetime import date

from stock_picker.log import get_logger
from stock_picker.paper.book import session_quotes_from_prices
from stock_picker.storage.paper_book_store import PaperBookStore, PaperPick
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.morning import score_from_quotes
from stock_picker.training.rank_model import RANK_TOP_K

logger = get_logger(__name__)


def replay_morning(
    as_of: date,
    model_run_id: str | None = None,
    threshold: float = 0.005,
    paper_store: PaperBookStore | None = None,
    price_store: PriceStore | None = None,
) -> str:
    """Score Fit+Rank on `as_of` using that day's daily-bar Open.

    Writes an extra paper-book scan (scan_id != '') so What if can show it
    next to the live morning without replacing it.
    """
    prices = price_store if price_store is not None else PriceStore()
    book = paper_store if paper_store is not None else PaperBookStore()
    tickers = UniverseStore().active_tickers()
    quotes = session_quotes_from_prices(tickers, as_of, prices)
    logger.info("replay %s quotes=%s model=%s", as_of, len(quotes), model_run_id or "live")
    freshness, rank_result, fit_result, _, _ = score_from_quotes(
        quotes,
        threshold=threshold,
        persist=False,
        news_fetcher=None,
        earnings_fetcher=lambda *a, **k: set(),
        as_of=as_of,
        model_run_id=model_run_id,
    )
    scan_id = uuid.uuid4().hex[:12]
    day = as_of.isoformat()
    picks: list[PaperPick] = []
    for kind, result in (("rank", rank_result), ("fit", fit_result)):
        if result is None:
            continue
        for rank, signal in enumerate(result.signals, start=1):
            if kind == "rank" and rank > RANK_TOP_K:
                break
            history = None
            try:
                history = prices.read(signal.ticker)
            except FileNotFoundError:
                history = None
            from stock_picker.paper.book import _bar, _prior_close, _session_from_bar

            _, close_px, _ = _session_from_bar(_bar(history, day) if history is not None else None)
            prev_close = _prior_close(history, day) if history is not None else None
            open_px = signal.open_price
            session_return = None
            if open_px and close_px and open_px > 0:
                session_return = (close_px / open_px) - 1.0
            picks.append(
                PaperPick(
                    as_of=day,
                    kind=kind,
                    rank=rank,
                    ticker=signal.ticker,
                    predicted=signal.predicted_return,
                    open_price=open_px,
                    close_price=close_px,
                    session_return=session_return,
                    news_flag=signal.news_flag,
                    news_checked=1,
                    prev_close=prev_close,
                    scan_id=scan_id,
                    model_run_id=model_run_id,
                )
            )
    book.replace_scan(day, scan_id, picks)
    logger.info("replay stored scan_id=%s picks=%s", scan_id, len(picks))
    return scan_id
