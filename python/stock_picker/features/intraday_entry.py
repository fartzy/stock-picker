"""Pure, shared historical/live features for entry-to-close predictions.

Massive minute ``t`` values identify bar starts. A bar contributes only after
``t + 60 seconds``; the current entry is a separately observed price. Neither
the current minute's final high/low/volume nor later bars enter these features.
https://massive.com/docs/rest/stocks/aggregates/custom-bars
"""
from __future__ import annotations

from datetime import datetime
import math
import statistics
from numbers import Real
from typing import Mapping, Sequence

MINUTE_MS = 60_000
FEATURE_VERSION = "intraday_entry_v1"
MORNING_FEATURE_COLUMNS = {
    "return_1d": "prior_return_1d",
    "return_2d": "prior_return_2d",
    "volatility_32d": "prior_volatility_32d",
    "rsi_28": "prior_rsi_28",
    "price_vs_sma_100d": "prior_price_vs_sma_100d",
    "volume_ratio_20d": "prior_volume_ratio_20d",
}
FEATURE_DESCRIPTIONS = {
    "morning_rank": "Original morning rank; a smaller number is a higher rank.",
    "morning_score_z": "Original score standardized within that morning's top 20 scores.",
    "minutes_since_open": "Minutes elapsed since the regular-session open.",
    "minutes_to_close": "Minutes remaining until this session's actual close.",
    "elapsed_session_fraction": "Fraction of this session elapsed, including early closes.",
    "return_from_open": "Current entry divided by this session's raw open, minus one.",
    "return_from_previous_close": "Current entry versus verified previous close; unknown across splits.",
    "opening_gap": "Raw session open versus verified previous close; unknown across splits.",
    "return_1m": "Change between the latest completed-minute close and the close one minute earlier.",
    "return_5m": "Completed-close change over five minutes; unknown if an intervening minute is missing.",
    "return_15m": "Completed-close change over 15 minutes; unknown if an intervening minute is missing.",
    "return_30m": "Completed-close change over 30 minutes; unknown if an intervening minute is missing.",
    "entry_vs_last_close": "Observed entry versus the last fully completed minute's close.",
    "distance_from_low_so_far": "Entry versus the lowest observed price so far, including the entry.",
    "drawdown_from_high_so_far": "Entry versus the highest observed price so far, including the entry.",
    "range_so_far": "Observed high-to-low range divided by the opening price.",
    "recovery_fraction_so_far": "Entry's location within the observed range so far.",
    "minutes_since_low": "Minutes since the latest observation of the running low.",
    "entry_vs_observed_vwap": "Entry versus volume-weighted VWAP of observed completed regular bars.",
    "log_observed_volume": "Natural log of one plus volume in observed completed regular bars.",
    "volume_acceleration_5m": "Mean volume over the last five minutes versus the preceding 15 minutes.",
    "realized_volatility_15m": "Population standard deviation of 15 consecutive completed log returns.",
    "observed_bar_fraction": "Share of expected completed regular minutes actually observed.",
    "previous_close_known": "One when a verified, same-price-basis previous close is available.",
    "prior_return_1d": "Existing one-day return feature known before this session.",
    "prior_return_2d": "Existing two-day return feature known before this session.",
    "prior_volatility_32d": "Existing 32-day volatility feature known before this session.",
    "prior_rsi_28": "Existing 28-day RSI feature known before this session.",
    "prior_price_vs_sma_100d": "Existing price versus 100-day average, known before this session.",
    "prior_volume_ratio_20d": "Existing 20-day volume ratio known before this session.",
    "return_from_open_in_prior_vol": "Entry's opening-price change divided by prior 32-day volatility.",
}
FEATURE_COLUMNS = tuple(FEATURE_DESCRIPTIONS)


def _finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value) -> bool:
    return _finite(value) and value > 0


def _stamp(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("intraday timestamps must include a timezone")
    return round(value.timestamp() * 1000)


def normalize_rank_score(score: float, top_scores: Sequence[float]) -> float:
    """Scale by the same top-20 cross-section in research and live inference."""
    values = [float(v) for v in top_scores if _finite(v)]
    if not _finite(score) or len(values) < 2:
        return float("nan")
    scale = statistics.pstdev(values)
    return (float(score) - statistics.fmean(values)) / scale if scale > 1e-12 else float("nan")


def build_intraday_entry_features(
    bars: Sequence[Mapping[str, float]], *, observed_at: datetime,
    session_open_at: datetime, session_close_at: datetime,
    entry_price: float, session_open: float,
    previous_close: float | None = None, rank: int | None = None,
    rank_score: float | None = None, rank_score_z: float | None = None,
    morning_features: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build only information available at ``observed_at``.

    ``morning_features`` must already be an as-of prior-session snapshot. The
    historical dataset uses the frozen shifted training rows; live callers must
    supply a strictly earlier-session FeatureStore row. ``rank_score`` is accepted
    for call-site provenance compatibility but deliberately is not a feature:
    different morning-model versions have different arbitrary raw-score scales.
    Sparse history is explicit through coverage and NaNs, never forward-filled.
    """
    del rank_score
    now, opened, closed = map(_stamp, (observed_at, session_open_at, session_close_at))
    if not opened + MINUTE_MS <= now < closed or closed <= opened:
        raise ValueError("entry must follow a completed minute within the regular session")
    if not _positive(entry_price) or not _positive(session_open):
        raise ValueError("entry and session opening prices must be finite and positive")
    completed: dict[int, Mapping[str, float]] = {}
    for bar in bars:
        t = bar.get("t")
        if not _finite(t) or t % MINUTE_MS:
            raise ValueError("invalid minute timestamp")
        if t < opened or t + MINUTE_MS > now:
            continue
        if t in completed:
            raise ValueError("duplicate minute timestamp")
        prices = [bar.get(k) for k in ("o", "h", "l", "c")]
        if not all(_positive(v) for v in prices) or not prices[2] <= min(prices[0], prices[3]) <= max(prices[0], prices[3]) <= prices[1]:
            raise ValueError("invalid completed-minute prices")
        completed[int(t)] = bar
    last = now // MINUTE_MS * MINUTE_MS - MINUTE_MS
    if last not in completed:
        raise ValueError("missing_last_completed_minute")
    ordered = sorted(completed)
    nan = float("nan")
    row = dict.fromkeys(FEATURE_COLUMNS, nan)
    elapsed = (now - opened) / MINUTE_MS
    prior_known = _positive(previous_close)
    row.update(
        morning_rank=float(rank) if _finite(rank) and rank >= 1 else nan,
        morning_score_z=float(rank_score_z) if _finite(rank_score_z) else nan,
        minutes_since_open=elapsed, minutes_to_close=(closed - now) / MINUTE_MS,
        elapsed_session_fraction=(now - opened) / (closed - opened),
        return_from_open=entry_price / session_open - 1,
        return_from_previous_close=entry_price / previous_close - 1 if prior_known else nan,
        opening_gap=session_open / previous_close - 1 if prior_known else nan,
        entry_vs_last_close=entry_price / completed[last]["c"] - 1,
        observed_bar_fraction=len(completed) / int((last + MINUTE_MS - opened) / MINUTE_MS),
        previous_close_known=float(prior_known),
    )
    def window(minutes: int):
        stamps = range(last - (minutes - 1) * MINUTE_MS, last + MINUTE_MS, MINUTE_MS)
        return [completed[t] for t in stamps] if all(t in completed for t in stamps) else None

    for minutes in (1, 5, 15, 30):
        values = window(minutes + 1)
        if values:
            row[f"return_{minutes}m"] = values[-1]["c"] / values[0]["c"] - 1
    low = min(session_open, entry_price, *(completed[t]["l"] for t in ordered))
    high = max(session_open, entry_price, *(completed[t]["h"] for t in ordered))
    low_observed = now if entry_price <= low else max(
        [opened, *(t + MINUTE_MS for t in ordered if completed[t]["l"] == low)])
    row.update(
        distance_from_low_so_far=entry_price / low - 1,
        drawdown_from_high_so_far=entry_price / high - 1,
        range_so_far=(high - low) / session_open,
        recovery_fraction_so_far=(entry_price - low) / (high - low) if high > low else 0.0,
        minutes_since_low=(now - low_observed) / MINUTE_MS,
    )
    volumes_known = all(_finite(completed[t].get("v")) and completed[t]["v"] >= 0 for t in ordered)
    if volumes_known:
        volume = sum(completed[t]["v"] for t in ordered)
        row["log_observed_volume"] = math.log1p(volume)
        if volume > 0 and all(completed[t]["v"] == 0 or _positive(completed[t].get("vw")) for t in ordered):
            vwap = sum(completed[t]["v"] * completed[t].get("vw", 0) for t in ordered if completed[t]["v"] > 0) / volume
            row["entry_vs_observed_vwap"] = entry_price / vwap - 1
    recent = window(20)
    if recent and all(_finite(b.get("v")) and b["v"] >= 0 for b in recent):
        previous_volume = statistics.fmean(b["v"] for b in recent[:-5])
        if previous_volume > 0:
            row["volume_acceleration_5m"] = statistics.fmean(b["v"] for b in recent[-5:]) / previous_volume - 1
    recent = window(16)
    if recent:
        row["realized_volatility_15m"] = statistics.pstdev(math.log(b["c"] / a["c"]) for a, b in zip(recent, recent[1:]))
    for source, target in MORNING_FEATURE_COLUMNS.items():
        value = (morning_features or {}).get(source)
        if _finite(value):
            row[target] = float(value)
    volatility = row["prior_volatility_32d"]
    if _positive(volatility):
        row["return_from_open_in_prior_vol"] = row["return_from_open"] / volatility
    return row
