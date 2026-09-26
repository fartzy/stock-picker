"""Rebuild and slice the paper book.

Every name on each morning list is stored. `paper_book_view` is the only
place Fit vs Rank and top-K are applied -- the SQLite table is the full
list so the What if tab can slice without another rebuild.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_picker.news_skip import news_blocks_buy
from stock_picker.storage.paper_book_store import PaperBookStore, PaperPick
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.scan_store import ScanStore

KINDS = ("fit", "rank")


def _bar(history: pd.DataFrame, day: str) -> pd.Series | None:
    if history.empty:
        return None
    index = pd.to_datetime(history.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    matched = history.loc[index == pd.Timestamp(day)]
    if matched.empty:
        return None
    return matched.iloc[-1]


def _prior_close(history: pd.DataFrame | None, day: str) -> float | None:
    """Yesterday's completed Close -- the gap vs this morning's open."""
    if history is None or history.empty or "Close" not in history.columns:
        return None
    index = pd.to_datetime(history.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    prior = history.loc[index < pd.Timestamp(day)]
    if prior.empty:
        return None
    close = prior.iloc[-1]["Close"]
    if pd.isna(close) or float(close) <= 0:
        return None
    return float(close)


def _session_from_bar(bar: pd.Series | None) -> tuple[float | None, float | None, float | None]:
    if bar is None or "Open" not in bar.index or "Close" not in bar.index:
        return None, None, None
    o, c = bar["Open"], bar["Close"]
    if pd.isna(o) or pd.isna(c) or float(o) <= 0:
        return None, None, None
    open_px, close_px = float(o), float(c)
    return open_px, close_px, (close_px / open_px) - 1.0


def session_quotes_from_prices(
    tickers: list[str],
    as_of: date,
    price_store: PriceStore | None = None,
) -> dict[str, dict]:
    """Official Open/Close/prev close from daily bars -- not a live snapshot.

    Historical What if must not call Yahoo v7. That path can copy yesterday's
    Open (WRBY $23.18). The bar for `as_of` is the session we are scoring.
    """
    prices = price_store if price_store is not None else PriceStore()
    day = as_of.isoformat()
    quotes: dict[str, dict] = {}
    for ticker in tickers:
        try:
            history = prices.read(ticker)
        except FileNotFoundError:
            continue
        open_px, close_px, _ = _session_from_bar(_bar(history, day))
        prev_close = _prior_close(history, day)
        if open_px is None or prev_close is None:
            continue
        quotes[ticker] = {
            "open": open_px,
            "last": close_px if close_px is not None else open_px,
            "prev_close": prev_close,
        }
    return quotes


def _quotes_for(tickers: list[str], as_of: date) -> dict[str, dict]:
    from stock_picker.ingestion.yfinance_client import fetch_quotes

    return fetch_quotes(tickers, as_of=as_of)


def _news_for_day(day: str, tickers: list[str], news_fetcher) -> dict[str, str]:
    if news_fetcher is None or not tickers:
        return {}
    try:
        return news_fetcher(tickers, date.fromisoformat(day)) or {}
    except TypeError:
        return {}


def rebuild_paper_book(
    completed_through: date | None = None,
    scan_store: ScanStore | None = None,
    price_store: PriceStore | None = None,
    paper_store: PaperBookStore | None = None,
    quote_fetcher=None,
    news_fetcher=None,
) -> list[PaperPick]:
    scans = scan_store if scan_store is not None else ScanStore()
    prices = price_store if price_store is not None else PriceStore()
    book = paper_store if paper_store is not None else PaperBookStore()
    quotes = quote_fetcher if quote_fetcher is not None else _quotes_for
    if completed_through is None:
        from stock_picker.ingestion.session import last_completed_session_date

        cutoff = last_completed_session_date()
    else:
        cutoff = completed_through

    picks: list[PaperPick] = []
    for day in scans.days():
        session_done = cutoff is None or date.fromisoformat(day) <= cutoff
        day_signals: list[tuple[str, list]] = []
        missing: list[str] = []
        for kind in KINDS:
            payload = scans.read(day, kind)
            if payload is None:
                continue
            signals = payload.get("signals") or []
            day_signals.append((kind, signals, payload.get("model_run_id")))
            if session_done:
                for signal in signals:
                    ticker = signal["ticker"]
                    try:
                        history = prices.read(ticker)
                    except FileNotFoundError:
                        history = None
                    if _session_from_bar(_bar(history, day) if history is not None else None)[0] is None:
                        missing.append(ticker)
        live = quotes(sorted(set(missing)), date.fromisoformat(day)) if missing else {}
        need_news: list[str] = []
        for _, signals, _ in day_signals:
            for signal in signals:
                if not signal.get("news_flag"):
                    need_news.append(signal["ticker"])
        unique_need = sorted(set(need_news))
        # Short lists only -- a 992-name Fit day would take minutes on Finnhub.
        from stock_picker.ingestion.finnhub_client import MAX_NEWS_TICKERS

        day_news_fetcher = news_fetcher
        if day_news_fetcher is None and 0 < len(unique_need) <= MAX_NEWS_TICKERS:
            from stock_picker.ingestion.finnhub_client import fetch_recent_news_flags

            day_news_fetcher = fetch_recent_news_flags
        fetched_news = _news_for_day(day, unique_need, day_news_fetcher)
        for kind, signals, model_run_id in day_signals:
            for rank, signal in enumerate(signals, start=1):
                ticker = signal["ticker"]
                predicted = signal.get("predicted_return")
                scan_open = signal.get("open_price")
                open_px = float(scan_open) if scan_open else None
                close_px = session_return = prev_close = None
                history = None
                if session_done:
                    try:
                        history = prices.read(ticker)
                    except FileNotFoundError:
                        history = None
                    bar_open, bar_close, _ = _session_from_bar(
                        _bar(history, day) if history is not None else None
                    )
                    prev_close = _prior_close(history, day)
                    if open_px is None:
                        open_px = bar_open
                    close_px = bar_close
                    if close_px is None and ticker in live:
                        quote = live[ticker]
                        if open_px is None and quote.get("open"):
                            open_px = float(quote["open"])
                        if quote.get("last"):
                            close_px = float(quote["last"])
                        if prev_close is None and quote.get("prev_close"):
                            prev_close = float(quote["prev_close"])
                    if open_px and close_px and open_px > 0:
                        session_return = (close_px / open_px) - 1.0
                cached_flag = signal.get("news_flag")
                news_flag = cached_flag or fetched_news.get(ticker)
                news_checked = 1 if (cached_flag is not None or ticker in fetched_news or news_fetcher is not None) else 0
                if cached_flag:
                    news_checked = 1
                elif fetched_news or news_fetcher is not None:
                    news_checked = 1
                picks.append(
                    PaperPick(
                        as_of=day,
                        kind=kind,
                        rank=rank,
                        ticker=ticker,
                        predicted=predicted,
                        open_price=open_px,
                        close_price=close_px,
                        session_return=session_return,
                        news_flag=news_flag,
                        news_checked=news_checked,
                        prev_close=prev_close,
                        scan_id="",
                        model_run_id=model_run_id,
                    )
                )
    book.replace_all(picks)
    return picks


def _scan_size(scans: ScanStore) -> int:
    total = 0
    for day in scans.days():
        for kind in KINDS:
            payload = scans.read(day, kind)
            if payload:
                total += len(payload.get("signals") or [])
    return total


def _missing_closes(picks: list[PaperPick], cutoff: date | None) -> bool:
    if cutoff is None:
        return False
    return any(
        p.session_return is None and date.fromisoformat(p.as_of) <= cutoff for p in picks
    )


def _news_unchecked(picks: list[PaperPick]) -> bool:
    return any(not p.news_checked for p in picks)


def _missing_prev_close_for_news(picks: list[PaperPick]) -> bool:
    return any(p.news_flag and p.prev_close is None for p in picks)


def load_paper_book(paper_store: PaperBookStore | None = None) -> list[PaperPick]:
    """SQLite only. Never Yahoo on a GET -- 990 names would hang What if."""
    book = paper_store if paper_store is not None else PaperBookStore()
    return book.read()


def refresh_paper_book(
    scan_store: ScanStore | None = None,
    price_store: PriceStore | None = None,
    paper_store: PaperBookStore | None = None,
    news_fetcher=None,
) -> list[PaperPick]:
    """Rebuild through the last completed session. Use the What if refresh button."""
    from stock_picker.ingestion.session import last_completed_session_date

    return rebuild_paper_book(
        completed_through=last_completed_session_date(),
        scan_store=scan_store,
        price_store=price_store,
        paper_store=paper_store,
        news_fetcher=news_fetcher,
    )


def _avg(rows: list[dict]) -> float | None:
    rets = [r["session_return"] for r in rows if r["session_return"] is not None]
    if not rets:
        return None
    return sum(rets) / len(rets)


def _list_stats(rows: list[dict]) -> dict:
    flagged = [r for r in rows if r.get("news_blocks")]
    kept = [r for r in rows if not r.get("news_blocks")]
    rets = [r["session_return"] for r in kept if r["session_return"] is not None]
    n_scored = len(rets)
    wins = sum(1 for r in rets if r > 0)
    losses = sum(1 for r in rets if r < 0)
    flats = n_scored - wins - losses
    avg = sum(rets) / n_scored if n_scored else None
    return {
        "n": len(rows),
        "n_scored": n_scored,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "hit_rate": (wins / n_scored) if n_scored else None,
        "avg": avg,
        "n_avoid": len(flagged),
        "avg_ex_news": avg,
    }


def _compound(avgs: list[float]) -> float | None:
    if not avgs:
        return None
    wealth = 1.0
    for avg in reversed(avgs):
        wealth *= 1.0 + avg
    return wealth - 1.0


def paper_book_view(
    picks: list[PaperPick],
    kind: str = "both",
    top_k: int | None = None,
    fit_top_k: int | None = None,
    rank_top_k: int | None = None,
) -> dict:
    """Slice stored picks. kind is fit, rank, or both. top_k None = all.

    fit_top_k / rank_top_k override top_k per list so Rank top 5 + Fit top 2
    does not need a rebuild.
    """
    wanted = KINDS if kind == "both" else (kind,)
    fit_k = fit_top_k if fit_top_k is not None else top_k
    rank_k = rank_top_k if rank_top_k is not None else top_k
    days: dict[tuple[str, str], dict] = {}
    for pick in picks:
        if pick.kind not in wanted:
            continue
        cap = fit_k if pick.kind == "fit" else rank_k
        if cap is not None and pick.rank > cap:
            continue
        scan_id = pick.scan_id or ""
        bucket = days.setdefault(
            (pick.as_of, scan_id),
            {
                "as_of": pick.as_of,
                "scan_id": scan_id,
                "model_run_id": pick.model_run_id,
                "fit": [],
                "rank": [],
            },
        )
        if pick.model_run_id and not bucket.get("model_run_id"):
            bucket["model_run_id"] = pick.model_run_id
        bucket[pick.kind].append(
            {
                "rank": pick.rank,
                "ticker": pick.ticker,
                "predicted": pick.predicted,
                "open_price": pick.open_price,
                "close_price": pick.close_price,
                "session_return": pick.session_return,
                "news_flag": pick.news_flag,
                "news_blocks": news_blocks_buy(pick.news_flag, pick.open_price, pick.prev_close),
            }
        )

    day_list = []
    fit_avgs: list[float] = []
    rank_avgs: list[float] = []
    fit_rows: list[dict] = []
    rank_rows: list[dict] = []
    for key in sorted(days, reverse=True):
        row = days[key]
        fit_stats = _list_stats(row["fit"])
        rank_stats = _list_stats(row["rank"])
        if not row["scan_id"]:
            if fit_stats["avg"] is not None:
                fit_avgs.append(fit_stats["avg"])
            if rank_stats["avg"] is not None:
                rank_avgs.append(rank_stats["avg"])
            fit_rows.extend(row["fit"])
            rank_rows.extend(row["rank"])
        day_list.append(
            {
                "as_of": row["as_of"],
                "scan_id": row["scan_id"],
                "model_run_id": row["model_run_id"],
                "fit": row["fit"],
                "rank": row["rank"],
                "fit_avg": fit_stats["avg"],
                "rank_avg": rank_stats["avg"],
                "fit_stats": fit_stats,
                "rank_stats": rank_stats,
            }
        )

    return {
        "days": day_list,
        "fit_compound": _compound(fit_avgs),
        "rank_compound": _compound(rank_avgs),
        "fit_days": len(fit_avgs),
        "rank_days": len(rank_avgs),
        "fit_stats": _list_stats(fit_rows),
        "rank_stats": _list_stats(rank_rows),
        "kind": kind,
        "top_k": top_k,
        "fit_top_k": fit_k,
        "rank_top_k": rank_k,
        "n_picks": sum(len(d["fit"]) + len(d["rank"]) for d in day_list),
    }
