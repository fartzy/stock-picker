"""Fake-open morning check: time Rank, Fit, and both on ~2000 names.

Quotes are invented from each ticker's last Close (labeled fake). No
Yahoo, no Finnhub -- this is a latency probe plus a look at which names
the live models would pick on those prints.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Literal

from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.buy_signal import DEFAULT_THRESHOLD, compute_buy_signals, compute_rank_signals
from stock_picker.training.rank_model import RANK_TOP_K

Status = Literal["idle", "running", "completed", "failed"]
Which = Literal["rank", "fit", "both"]


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


def fake_quotes_from_last_close(
    universe: UniverseStore | None = None,
    prices: PriceStore | None = None,
) -> list[FakeQuote]:
    """Last Close is today's fake open. Missing parquet -> skip that ticker."""
    universe = universe or UniverseStore()
    prices = prices or PriceStore()
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
        out.append(FakeQuote(ticker=ticker, fake_open=last_close, last_close=last_close))
    return out


def _quote_map(quotes: list[FakeQuote]) -> dict[str, dict]:
    return {
        q.ticker: {"open": q.fake_open, "last": q.fake_open, "prev_close": q.last_close}
        for q in quotes
    }


def _pick_rows(result) -> list[dict]:
    return [
        {
            "ticker": s.ticker,
            "predicted_return": s.predicted_return,
            "open_price": s.open_price,
            "snapshot_date": s.snapshot_date,
        }
        for s in result.signals
    ]


def run_morning_check(
    which: Which = "both",
    universe: UniverseStore | None = None,
    prices: PriceStore | None = None,
) -> MorningCheckStatus:
    started = datetime.now().astimezone().isoformat()
    t0 = time.perf_counter()
    quotes = fake_quotes_from_last_close(universe=universe, prices=prices)
    quote_seconds = time.perf_counter() - t0
    quote_lookup = _quote_map(quotes)

    def fetch(_tickers, as_of=None):
        return quote_lookup

    def _one(name: str) -> TimedPass:
        t1 = time.perf_counter()
        if name == "rank":
            result = compute_rank_signals(
                top_k=RANK_TOP_K,
                quote_fetcher=fetch,
                earnings_fetcher=None,
                news_fetcher=None,
            )
        else:
            result = compute_buy_signals(
                threshold=DEFAULT_THRESHOLD,
                quote_fetcher=fetch,
                earnings_fetcher=None,
                news_fetcher=None,
            )
        return TimedPass(
            which=name,
            seconds=time.perf_counter() - t1,
            scored_count=result.scored_count,
            n_picks=len(result.signals),
            picks=_pick_rows(result),
            skipped_count=len(result.skipped),
        )

    wanted = ("rank", "fit") if which == "both" else (which,)
    wall_t0 = time.perf_counter()
    if len(wanted) == 1:
        passes = [_one(wanted[0])]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_one, name): name for name in wanted}
            by_name = {futures[fut]: fut.result() for fut in as_completed(futures)}
        passes = [by_name["rank"], by_name["fit"]]
        wall = time.perf_counter() - wall_t0
        passes.append(
            TimedPass(
                which="both",
                seconds=wall,
                scored_count=max(p.scored_count for p in passes),
                n_picks=sum(p.n_picks for p in passes),
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
