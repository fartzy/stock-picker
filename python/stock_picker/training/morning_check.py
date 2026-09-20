"""Test run: only the opens are fake. Everything else is the real morning path.

Fake opens = last Close jittered by a small random gap. Rank + Fit run in
parallel, earnings skip and news on Fit + Rank top 10 -- same as 8:32 CT.
persist=False so this does not overwrite Monday's cache.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.buy_signal import DEFAULT_THRESHOLD
from stock_picker.training.morning import score_from_quotes
from stock_picker.training.news_day_judge import fetch_recent_news_flags


Status = Literal["idle", "running", "completed", "failed"]
Which = Literal["rank", "fit", "both"]

# Uniform gap around last close so open-known features actually move.
FAKE_GAP_RANGE = (-0.03, 0.03)


@dataclass
class FakeQuote:
    ticker: str
    fake_open: float
    last_close: float


@dataclass
class TimedPass:
    which: str
    seconds: float
    scored_count: int
    n_picks: int
    picks: list[dict] = field(default_factory=list)
    skipped_count: int = 0


@dataclass
class MorningCheckStatus:
    status: Status = "idle"
    which: Which | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    n_quotes: int = 0
    quote_seconds: float | None = None
    quotes: list[FakeQuote] = field(default_factory=list)
    passes: list[TimedPass] = field(default_factory=list)


def randomized_fake_quotes(
    universe: UniverseStore | None = None,
    prices: PriceStore | None = None,
    rng: random.Random | None = None,
) -> list[FakeQuote]:
    """Last Close * (1 + U(-3%, +3%)). Missing parquet -> skip."""
    universe = universe or UniverseStore()
    prices = prices or PriceStore()
    rng = rng or random.Random()
    lo, hi = FAKE_GAP_RANGE
    out: list[FakeQuote] = []
    for ticker in universe.active_tickers():
        try:
            history = prices.read(ticker)
        except FileNotFoundError:
            continue
        if history.empty or "Close" not in history.columns:
            continue
        close = history["Close"].iloc[-1]
        if close != close or float(close) <= 0:
            continue
        last_close = float(close)
        fake_open = round(last_close * (1.0 + rng.uniform(lo, hi)), 4)
        if fake_open <= 0:
            fake_open = last_close
        out.append(FakeQuote(ticker=ticker, fake_open=fake_open, last_close=last_close))
    return out


def _quote_map(quotes: list[FakeQuote]) -> dict[str, dict]:
    return {
        q.ticker: {"open": q.fake_open, "last": q.fake_open, "prev_close": q.last_close}
        for q in quotes
    }


def _pick_rows(signals) -> list[dict]:
    return [
        {
            "ticker": s.ticker,
            "predicted_return": s.predicted_return,
            "open_price": s.open_price,
            "snapshot_date": s.snapshot_date,
            "news_flag": s.news_flag,
        }
        for s in signals
    ]


def run_morning_check(
    which: Which = "both",
    universe: UniverseStore | None = None,
    prices: PriceStore | None = None,
    news_fetcher=fetch_recent_news_flags,
) -> MorningCheckStatus:
    started = datetime.now().astimezone().isoformat()
    t0 = time.perf_counter()
    quotes = randomized_fake_quotes(universe=universe, prices=prices)
    quote_seconds = time.perf_counter() - t0
    quote_lookup = _quote_map(quotes)

    wall_t0 = time.perf_counter()
    _freshness, rank_result, fit_result, _rank_n, _rank_text = score_from_quotes(
        quote_lookup,
        threshold=DEFAULT_THRESHOLD,
        persist=False,
        news_fetcher=news_fetcher if which in ("fit", "both", "rank") else None,
    )
    wall = time.perf_counter() - wall_t0

    passes: list[TimedPass] = []
    if which in ("rank", "both") and rank_result is not None:
        passes.append(
            TimedPass(
                which="rank",
                seconds=wall if which == "rank" else wall,
                scored_count=rank_result.scored_count,
                n_picks=len(rank_result.signals),
                picks=_pick_rows(rank_result.signals),
                skipped_count=len(rank_result.skipped),
            )
        )
    if which in ("fit", "both"):
        passes.append(
            TimedPass(
                which="fit",
                seconds=wall if which == "fit" else wall,
                scored_count=fit_result.scored_count,
                n_picks=len(fit_result.signals),
                picks=_pick_rows(fit_result.signals),
                skipped_count=len(fit_result.skipped),
            )
        )
    if which == "both":
        passes.append(
            TimedPass(
                which="both",
                seconds=wall,
                scored_count=max(fit_result.scored_count, rank_result.scored_count if rank_result else 0),
                n_picks=sum(p.n_picks for p in passes if p.which != "both"),
            )
        )

    return MorningCheckStatus(
        status="completed",
        which=which,
        started_at=started,
        completed_at=datetime.now().astimezone().isoformat(),
        n_quotes=len(quotes),
        quote_seconds=quote_seconds,
        quotes=quotes,
        passes=passes,
    )


class MorningCheckJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = MorningCheckStatus()

    def status(self) -> MorningCheckStatus:
        with self._lock:
            return replace(self._state)

    def start(self, which: Which = "both") -> bool:
        with self._lock:
            if self._state.status == "running":
                return False
            self._state = MorningCheckStatus(
                status="running",
                which=which,
                started_at=datetime.now().astimezone().isoformat(),
            )

        def _run() -> None:
            try:
                result = run_morning_check(which=which)
                with self._lock:
                    self._state = result
            except Exception as exc:  # noqa: BLE001 -- surfaced on the panel
                with self._lock:
                    self._state = replace(
                        self._state,
                        status="failed",
                        completed_at=datetime.now().astimezone().isoformat(),
                        error=str(exc),
                    )

        threading.Thread(target=_run, daemon=True).start()
        return True


_default = MorningCheckJob()


def status() -> MorningCheckStatus:
    return _default.status()


def start(which: Which = "both") -> bool:
    return _default.start(which)
