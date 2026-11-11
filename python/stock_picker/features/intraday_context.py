"""Experimental current-entry context, shared by offline and future live callers.

No network or target data enters these calculations. Prior sessions must be
selected by a real exchange calendar, not by whichever responses succeeded.
"""
from __future__ import annotations

from datetime import datetime
import math
import statistics
from typing import Mapping, Sequence

Bar = Mapping[str, float]
MARKET_DESCRIPTIONS = {
    "spy_return_from_open": "SPY completed close versus first regular minute open.",
    "spy_return_15m": "SPY completed-close return over fifteen consecutive minutes.",
    "stock_minus_spy_from_open": "Stock entry return from open minus observed SPY return from open; unit-beta difference.",
    "stock_minus_spy_15m": "Stock completed fifteen-minute return minus SPY equivalent.",
}
VOLUME_DESCRIPTIONS = {
    "log_same_clock_dollar_volume": "Log cumulative dollar-volume ratio to the median at this clock on prior twenty sessions; at least ten references.",
    "log_same_clock_5m_dollar_volume": "Log five-minute dollar-volume ratio to the median matching interval on prior twenty sessions; at least ten references.",
}
CONTEXT_DESCRIPTIONS = {**MARKET_DESCRIPTIONS, **VOLUME_DESCRIPTIONS}
MINUTE = 60_000
MIN_HISTORY = 10
MIN_COVERAGE = .8


def _stamp(instant: datetime) -> int:
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("context timestamps require timezone")
    return round(instant.timestamp() * 1000)


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def completed(bars: Sequence[Bar], opened: datetime, observed: datetime) -> dict[int, Bar]:
    start, end = _stamp(opened), _stamp(observed)
    selected = [bar for bar in bars if start <= bar["t"] and bar["t"] + MINUTE <= end]
    result = {int(bar["t"]): bar for bar in selected}
    if len(result) != len(selected):
        raise ValueError("duplicate context bar timestamps")
    if any(stamp % MINUTE for stamp in result):
        raise ValueError("unaligned context minute")
    return result


def dollar_volume(bars: Mapping[int, Bar], start: int, end: int) -> float:
    """Unknown when sparse/missing VWAP; absent trades are not fabricated bars."""
    expected = (end - start) // MINUTE
    selected = [bar for stamp, bar in bars.items() if start <= stamp < end]
    if expected <= 0 or len(selected) / expected < MIN_COVERAGE:
        return math.nan
    if any(not _positive(bar.get("vw")) or not isinstance(bar.get("v"), (int, float))
           or not math.isfinite(bar["v"]) or bar["v"] < 0 for bar in selected):
        return math.nan
    return sum(bar["v"] * bar["vw"] for bar in selected)


def build_intraday_context(
    *, stock_bars: Sequence[Bar], spy_bars: Sequence[Bar],
    history: Sequence[tuple[datetime, datetime, Sequence[Bar]]],
    observed_at: datetime, session_open_at: datetime,
    stock_return_from_open: float, stock_return_15m: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Use only complete bars and strictly earlier, externally selected sessions.

    ``history`` contains up to the previous twenty exchange sessions, including
    empty/failed days. They cannot be replaced with older successful downloads.
    Same elapsed minutes match ET clock on normal/early-close sessions and DST.
    """
    now, opened = _stamp(observed_at), _stamp(session_open_at)
    if now <= opened or now % MINUTE:
        raise ValueError("use an aligned completed-minute decision after open")
    if len(history) > 20:
        raise ValueError("at most twenty prior calendar sessions")
    dates = [_stamp(start) for start, _, _ in history]
    if len(set(dates)) != len(dates) or any(_stamp(end) >= opened for _, end, _ in history):
        raise ValueError("history must contain unique strictly prior sessions")
    values = dict.fromkeys(CONTEXT_DESCRIPTIONS, math.nan)
    spy = completed(spy_bars, session_open_at, observed_at)
    latest, first = spy.get(now - MINUTE), spy.get(opened)
    if latest and first and _positive(first.get("o")) and _positive(latest.get("c")):
        values["spy_return_from_open"] = latest["c"] / first["o"] - 1
    past = spy.get(now - 16 * MINUTE)
    if latest and past and _positive(latest.get("c")) and _positive(past.get("c")) and all(
        now - offset * MINUTE in spy for offset in range(1, 17)
    ):
        values["spy_return_15m"] = latest["c"] / past["c"] - 1
    values["stock_minus_spy_from_open"] = stock_return_from_open - values["spy_return_from_open"]
    values["stock_minus_spy_15m"] = stock_return_15m - values["spy_return_15m"]
    stock = completed(stock_bars, session_open_at, observed_at)
    elapsed = observed_at - session_open_at
    samples = {"cumulative": [], "5m": []}
    for start, end, bars in history:
        checkpoint = start + elapsed
        if checkpoint >= end:
            continue
        prior = completed(bars, start, checkpoint)
        for key, lower in (("cumulative", _stamp(start)), ("5m", _stamp(checkpoint) - 5 * MINUTE)):
            if lower < _stamp(start):
                continue
            value = dollar_volume(prior, lower, _stamp(checkpoint))
            if _positive(value):
                samples[key].append(value)
    for key, feature, lower in (
        ("cumulative", "log_same_clock_dollar_volume", opened),
        ("5m", "log_same_clock_5m_dollar_volume", now - 5 * MINUTE),
    ):
        current = dollar_volume(stock, lower, now) if lower >= opened else math.nan
        if len(samples[key]) >= MIN_HISTORY and _positive(current):
            values[feature] = math.log(current / statistics.median(samples[key]))
    return values, {key + "_reference_sessions": len(value) for key, value in samples.items()}
