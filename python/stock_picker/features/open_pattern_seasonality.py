"""Open-known recency patterns: last few completed days + this morning's open,
scored against all prior history.

Each column is the expanding mean of prior same-bucket day-session (open->close)
returns -- "the last times this ticker looked like this at the open, what did
the rest of that day do?" The bucket always includes today's open (gap vs
yesterday's close, or where that open sits in yesterday's range), so these
are knowable at 9:30 and must NOT be features.shift(1)'d. See
training/dataset.py's OPEN_KNOWN_FEATURE_COLUMNS and inference.py, which
recomputes them from yesterday's OHLCV + this morning's print.

Completed-day path (U/D sequence, streak, vol) never includes today's close;
today only contributes the open. prior_bucket_mean shifts within the bucket
so the current row's own session return is not in its own feature value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_picker.features.candle import overnight_gap
from stock_picker.features.conditional_seasonality import (
    GAP_FLAT_THRESHOLD,
    bucket_signal,
)
from stock_picker.features.pattern_seasonality import day_session_return, day_session_streak
from stock_picker.features.volatility import ATR_WINDOW, average_true_range

GAP_LARGE_THRESHOLD = 0.015
VOL_WINDOW = 20
WILD_QUIET = 0.8
WILD_HIGH = 1.5
RANGE_NARROW = 0.7
RANGE_WIDE = 1.3
VOL_REGIME_LOW = 0.8
VOL_REGIME_HIGH = 1.2
STREAK_CAP = 3
CRASH_ATR = 2.0
EXTREME_ATR = 2.5

OPEN_KNOWN_COLUMNS = [
    "seq2_open3_seasonality",
    "seq3_open3_seasonality",
    "seq4_open3_seasonality",
    "seq3_open5_seasonality",
    "streak_open3_seasonality",
    "streak_open5_seasonality",
    "flips_3d_open3_seasonality",
    "flips_5d_open3_seasonality",
    "late_reversal_open3_seasonality",
    "continuation_open3_seasonality",
    "yday_swing_open5_seasonality",
    "two_day_swing_open3_seasonality",
    "big_down_run_open3_seasonality",
    "net3_swing_open3_seasonality",
    "wildest_3d_open3_seasonality",
    "vol_path_open3_seasonality",
    "vol_path_open5_seasonality",
    "yday_range_open3_seasonality",
    "vol_regime_open3_seasonality",
    "seq3_volpath_open3_seasonality",
    "swing_volpath_open3_seasonality",
    "streak_vol_open3_seasonality",
    "open_in_yday_range_seasonality",
    "seq3_openloc_seasonality",
    "gap_seq3_open3_seasonality",
    "gap_trap_open_seasonality",
    "streak_crash_open_seasonality",
    "multi_crash_bounce_open_seasonality",
]


def prior_bucket_mean(values: pd.Series, bucket: pd.Series) -> pd.Series:
    """Expanding mean of `values` within `bucket`, using only strictly prior
    occurrences -- the current row's own value is not in its own average."""
    return values.groupby(bucket).transform(lambda s: s.expanding().mean().shift(1))


def combine_keys(*parts: pd.Series, sep: str = "_") -> pd.Series:
    """Join bucket parts into one key; NaN in any part propagates."""
    result = pd.Series(pd.NA, index=parts[0].index, dtype="object")
    valid = parts[0].notna()
    for part in parts[1:]:
        valid &= part.notna()
    if valid.any():
        result.loc[valid] = [
            sep.join(str(item) for item in row) for row in zip(*(part[valid] for part in parts))
        ]
    return result


def session_sign(session: pd.Series) -> pd.Series:
    """+1 if day-session return >= 0, -1 if down. NaN stays NaN (never 0)."""
    sign = pd.Series(np.nan, index=session.index, dtype="float64")
    valid = session.notna()
    sign[valid] = np.where(session[valid] >= 0, 1.0, -1.0)
    return sign


def direction_letter(session: pd.Series) -> pd.Series:
    letters = pd.Series(pd.NA, index=session.index, dtype="object")
    sign = session_sign(session)
    letters[sign == 1.0] = "U"
    letters[sign == -1.0] = "D"
    return letters


def completed_sequence(session: pd.Series, n: int) -> pd.Series:
    """Exact U/D path of the last `n` completed day-sessions, oldest first.

    At row t (this morning) this is t-n .. t-1, never today.
    """
    letters = [direction_letter(session.shift(i)) for i in range(n, 0, -1)]
    return combine_keys(*letters, sep="")


def bucket3_label(signal: pd.Series, flat_threshold: float = GAP_FLAT_THRESHOLD) -> pd.Series:
    raw = bucket_signal(signal, flat_threshold)
    labels = pd.Series(pd.NA, index=signal.index, dtype="object")
    labels[raw == -1] = "down"
    labels[raw == 0] = "flat"
    labels[raw == 1] = "up"
    return labels


def bucket5_gap(
    gap: pd.Series,
    flat_threshold: float = GAP_FLAT_THRESHOLD,
    large_threshold: float = GAP_LARGE_THRESHOLD,
) -> pd.Series:
    labels = pd.Series(pd.NA, index=gap.index, dtype="object")
    valid = gap.notna()
    labels[valid & (gap < -large_threshold)] = "way_down"
    labels[valid & (gap >= -large_threshold) & (gap < -flat_threshold)] = "down"
    labels[valid & (gap.abs() <= flat_threshold)] = "flat"
    labels[valid & (gap > flat_threshold) & (gap <= large_threshold)] = "up"
    labels[valid & (gap > large_threshold)] = "way_up"
    return labels


def open_in_yday_range(history: pd.DataFrame) -> pd.Series:
    """Where today's open sits vs yesterday's high/low: below / low / mid / high / above."""
    yday_low = history["Low"].shift(1)
    yday_high = history["High"].shift(1)
    open_ = history["Open"]
    labels = pd.Series(pd.NA, index=history.index, dtype="object")
    span = yday_high - yday_low
    valid = yday_low.notna() & yday_high.notna() & open_.notna() & (span > 0)
    labels[valid & (open_ < yday_low)] = "below"
    labels[valid & (open_ > yday_high)] = "above"
    loc = (open_ - yday_low) / span
    inside = valid & (open_ >= yday_low) & (open_ <= yday_high)
    labels[inside & (loc < 1 / 3)] = "low"
    labels[inside & (loc >= 1 / 3) & (loc <= 2 / 3)] = "mid"
    labels[inside & (loc > 2 / 3)] = "high"
    return labels


def open_loc3(location5: pd.Series) -> pd.Series:
    labels = pd.Series(pd.NA, index=location5.index, dtype="object")
    labels[location5.isin(["below", "low"])] = "low"
    labels[location5 == "mid"] = "mid"
    labels[location5.isin(["high", "above"])] = "high"
    return labels


def flip_count(*signs: pd.Series, cap: int | None = None) -> pd.Series:
    valid = signs[0].notna()
    for sign in signs[1:]:
        valid &= sign.notna()
    flips = pd.Series(np.nan, index=signs[0].index, dtype="float64")
    acc = pd.Series(0.0, index=signs[0].index)
    for left, right in zip(signs, signs[1:]):
        acc = acc + (left != right).astype(float)
    if cap is not None:
        acc = acc.clip(upper=cap)
    flips[valid] = acc[valid]
    return flips


def large_down_run(large_down: pd.Series) -> pd.Series:
    """Consecutive large-down completed days, capped at 2, as of yesterday."""
    run = pd.Series(0.0, index=large_down.index)
    current = 0.0
    for i, flag in enumerate(large_down.to_numpy()):
        if pd.isna(flag):
            current = 0.0
        elif flag:
            current = min(current + 1.0, 2.0)
        else:
            current = 0.0
        run.iloc[i] = current
    completed = run.shift(1)
    labels = pd.Series(pd.NA, index=large_down.index, dtype="object")
    valid = completed.notna()
    labels[valid] = completed[valid].astype(int).astype(str)
    return labels


def expanding_contracting(delta: pd.Series) -> pd.Series:
    labels = pd.Series(pd.NA, index=delta.index, dtype="object")
    valid = delta.notna()
    labels[valid & (delta > 0)] = "exp"
    labels[valid & (delta <= 0)] = "con"
    return labels


def size_label(is_large: pd.Series) -> pd.Series:
    labels = pd.Series(pd.NA, index=is_large.index, dtype="object")
    labels[is_large.eq(True)] = "large"
    labels[is_large.eq(False)] = "small"
    return labels


@dataclass
class _Parts:
    session: pd.Series
    open3: pd.Series
    open5: pd.Series
    loc5: pd.Series
    loc3: pd.Series
    seq2: pd.Series
    seq3: pd.Series
    seq4: pd.Series
    streak: pd.Series
    streak_sign: pd.Series
    flips3: pd.Series
    flips5: pd.Series
    late_reversal: pd.Series
    continuation: pd.Series
    yday_swing: pd.Series
    two_day_swing: pd.Series
    down_run: pd.Series
    net3: pd.Series
    wildness: pd.Series
    wildest: pd.Series
    vol_path: pd.Series
    yday_range: pd.Series
    vol_regime: pd.Series
    net3_dir: pd.Series
    yday_big: pd.Series
    vol_3d: pd.Series
    gap_seq3: pd.Series
    gap_trap: pd.Series
    streak_crash: pd.Series
    multi_crash: pd.Series


def _parts(history: pd.DataFrame) -> _Parts:
    session = day_session_return(history)
    gap = overnight_gap(history["Open"], history["Close"])
    open3 = bucket3_label(gap)
    open5 = bucket5_gap(gap)
    loc5 = open_in_yday_range(history)
    loc3 = open_loc3(loc5)

    s1 = session_sign(session.shift(1))
    s2 = session_sign(session.shift(2))
    s3 = session_sign(session.shift(3))
    s4 = session_sign(session.shift(4))
    s5 = session_sign(session.shift(5))

    streak = day_session_streak(session, cap=STREAK_CAP).shift(1)
    streak_key = pd.Series(pd.NA, index=history.index, dtype="object")
    streak_valid = streak.notna()
    streak_key[streak_valid] = streak[streak_valid].astype(int).astype(str)
    streak_sign = bucket3_label(streak, flat_threshold=0.5)

    flips3 = flip_count(s1, s2, s3)
    flips3_key = pd.Series(pd.NA, index=history.index, dtype="object")
    flips3_valid = flips3.notna()
    flips3_key[flips3_valid] = flips3[flips3_valid].astype(int).astype(str)

    flips5 = flip_count(s1, s2, s3, s4, s5, cap=2)
    flips5_key = pd.Series(pd.NA, index=history.index, dtype="object")
    flips5_valid = flips5.notna()
    flips5_key[flips5_valid] = flips5[flips5_valid].astype(int).astype(str)

    majority = s1 + s2 + s3
    majority_sign = pd.Series(np.nan, index=history.index, dtype="float64")
    majority_valid = s1.notna() & s2.notna() & s3.notna()
    majority_sign[majority_valid] = np.sign(majority[majority_valid])
    reversal = pd.Series(pd.NA, index=history.index, dtype="object")
    reversal[majority_valid & (majority_sign != 0) & (s1 != majority_sign)] = "reversal"
    reversal[majority_valid & ~((majority_sign != 0) & (s1 != majority_sign))] = "no_reversal"

    have_three = s1.notna() & s2.notna() & s3.notna()
    all_same = have_three & (s1 == s2) & (s2 == s3)
    last_dir = direction_letter(session.shift(1))
    continuation_kind = pd.Series(pd.NA, index=history.index, dtype="object")
    continuation_kind[all_same] = "same"
    continuation_kind[have_three & ~all_same] = "mixed"
    continuation = combine_keys(continuation_kind, last_dir)

    daily_vol = history["Close"].pct_change().rolling(VOL_WINDOW).std()
    yday = session.shift(1)
    two_ago = session.shift(2)
    yday_vol = daily_vol.shift(1)
    two_ago_vol = daily_vol.shift(2)
    yday_large = (yday.abs() > yday_vol).where(yday.notna() & yday_vol.notna())
    two_large = (two_ago.abs() > two_ago_vol).where(two_ago.notna() & two_ago_vol.notna())

    yday_swing = pd.Series(pd.NA, index=history.index, dtype="object")
    yday_valid = yday.notna() & yday_vol.notna()
    yday_swing[yday_valid & (yday < 0) & yday_large.eq(True)] = "big_down"
    yday_swing[yday_valid & (yday < 0) & yday_large.eq(False)] = "down"
    yday_swing[yday_valid & (yday >= 0) & yday_large.eq(False)] = "up"
    yday_swing[yday_valid & (yday >= 0) & yday_large.eq(True)] = "big_up"

    two_day_swing = combine_keys(
        size_label(two_large),
        size_label(yday_large),
        last_dir,
    )

    large_down = (session < 0) & (session.abs() > daily_vol)
    down_run = large_down_run(large_down)

    net3 = session.shift(1) + session.shift(2) + session.shift(3)
    net3_label = bucket3_label(net3)
    net3_dir = direction_letter(net3)
    abs3 = pd.concat(
        [session.shift(1).abs(), session.shift(2).abs(), session.shift(3).abs()], axis=1
    ).max(axis=1)
    wild_ratio = abs3 / yday_vol
    wildness = pd.Series(pd.NA, index=history.index, dtype="object")
    wild_valid = wild_ratio.notna()
    wildness[wild_valid & (wild_ratio < WILD_QUIET)] = "quiet"
    wildness[wild_valid & (wild_ratio >= WILD_QUIET) & (wild_ratio <= WILD_HIGH)] = "normal"
    wildness[wild_valid & (wild_ratio > WILD_HIGH)] = "wild"

    wildest = pd.Series(pd.NA, index=history.index, dtype="object")
    wildest[wild_valid & (wild_ratio < WILD_QUIET)] = "quiet"
    wildest[wild_valid & (wild_ratio >= WILD_QUIET) & (wild_ratio <= WILD_HIGH)] = "normal"
    wildest[wild_valid & (wild_ratio > WILD_HIGH)] = "wild"

    vol_3d_delta = daily_vol.shift(1) - daily_vol.shift(4)
    vol_yday_delta = daily_vol.shift(1) - daily_vol.shift(2)
    vol_3d = expanding_contracting(vol_3d_delta)
    vol_path = combine_keys(vol_3d, expanding_contracting(vol_yday_delta))

    atr = average_true_range(history, window=ATR_WINDOW)
    yday_range_ratio = (history["High"] - history["Low"]).shift(1) / atr.shift(1)
    yday_range = pd.Series(pd.NA, index=history.index, dtype="object")
    range_valid = yday_range_ratio.notna()
    yday_range[range_valid & (yday_range_ratio < RANGE_NARROW)] = "narrow"
    yday_range[range_valid & (yday_range_ratio >= RANGE_NARROW) & (yday_range_ratio <= RANGE_WIDE)] = (
        "normal"
    )
    yday_range[range_valid & (yday_range_ratio > RANGE_WIDE)] = "wide"

    vol_vs_median = daily_vol.shift(1) / daily_vol.shift(1).rolling(60, min_periods=20).median()
    vol_regime = pd.Series(pd.NA, index=history.index, dtype="object")
    regime_valid = vol_vs_median.notna()
    vol_regime[regime_valid & (vol_vs_median < VOL_REGIME_LOW)] = "low"
    vol_regime[regime_valid & (vol_vs_median >= VOL_REGIME_LOW) & (vol_vs_median <= VOL_REGIME_HIGH)] = (
        "mid"
    )
    vol_regime[regime_valid & (vol_vs_median > VOL_REGIME_HIGH)] = "high"

    yday_big = pd.Series(pd.NA, index=history.index, dtype="object")
    yday_big[yday_valid] = np.where(yday_large[yday_valid], "big", "not_big")

    gap_seq3 = combine_keys(bucket3_label(gap.shift(2)), bucket3_label(gap.shift(1)), open3)

    yday_gap_ud = pd.Series(pd.NA, index=history.index, dtype="object")
    yday_gap = gap.shift(1)
    yday_gap_ud[yday_gap > GAP_FLAT_THRESHOLD] = "up"
    yday_gap_ud[yday_gap < -GAP_FLAT_THRESHOLD] = "down"
    gap_trap = combine_keys(yday_gap_ud, last_dir, open3)

    session_atr = (history["Close"] - history["Open"]) / atr.replace(0, np.nan)
    into_streak = day_session_streak(session, cap=STREAK_CAP).shift(2)
    into_label = pd.Series(pd.NA, index=history.index, dtype="object")
    into_label[into_streak == STREAK_CAP] = "up3"
    into_label[into_streak == -STREAK_CAP] = "down3"
    into_label[into_streak.notna() & (into_streak.abs() < STREAK_CAP)] = "other"
    yday_crash = session_atr.shift(1) < -CRASH_ATR
    crash_label = pd.Series(pd.NA, index=history.index, dtype="object")
    crash_label[yday_crash.eq(True)] = "crash"
    crash_label[yday_crash.eq(False)] = "no_crash"
    streak_crash = combine_keys(into_label, crash_label, open3)

    all_down_4 = (s1 == -1.0) & (s2 == -1.0) & (s3 == -1.0) & (s4 == -1.0)
    have_four = s1.notna() & s2.notna() & s3.notna() & s4.notna()
    down_path = pd.Series(pd.NA, index=history.index, dtype="object")
    down_path[have_four] = "not_dddd"
    down_path[all_down_4] = "dddd"
    yday_extreme = session_atr.shift(1).abs() > EXTREME_ATR
    extreme_label = pd.Series(pd.NA, index=history.index, dtype="object")
    extreme_label[yday_extreme.eq(True)] = "extreme"
    extreme_label[yday_extreme.eq(False)] = "normal"
    multi_crash = combine_keys(down_path, extreme_label, open5)

    return _Parts(
        session=session,
        open3=open3,
        open5=open5,
        loc5=loc5,
        loc3=loc3,
        seq2=completed_sequence(session, 2),
        seq3=completed_sequence(session, 3),
        seq4=completed_sequence(session, 4),
        streak=streak_key,
        streak_sign=streak_sign,
        flips3=flips3_key,
        flips5=flips5_key,
        late_reversal=reversal,
        continuation=continuation,
        yday_swing=yday_swing,
        two_day_swing=two_day_swing,
        down_run=down_run,
        net3=net3_label,
        wildness=wildness,
        wildest=wildest,
        vol_path=vol_path,
        yday_range=yday_range,
        vol_regime=vol_regime,
        net3_dir=net3_dir,
        yday_big=yday_big,
        vol_3d=vol_3d,
        gap_seq3=gap_seq3,
        gap_trap=gap_trap,
        streak_crash=streak_crash,
        multi_crash=multi_crash,
    )


def build_open_pattern_features(history: pd.DataFrame) -> pd.DataFrame:
    parts = _parts(history)
    buckets = {
        "seq2_open3_seasonality": combine_keys(parts.seq2, parts.open3),
        "seq3_open3_seasonality": combine_keys(parts.seq3, parts.open3),
        "seq4_open3_seasonality": combine_keys(parts.seq4, parts.open3),
        "seq3_open5_seasonality": combine_keys(parts.seq3, parts.open5),
        "streak_open3_seasonality": combine_keys(parts.streak, parts.open3),
        "streak_open5_seasonality": combine_keys(parts.streak, parts.open5),
        "flips_3d_open3_seasonality": combine_keys(parts.flips3, direction_letter(parts.session.shift(1)), parts.open3),
        "flips_5d_open3_seasonality": combine_keys(parts.flips5, direction_letter(parts.session.shift(1)), parts.open3),
        "late_reversal_open3_seasonality": combine_keys(parts.late_reversal, parts.open3),
        "continuation_open3_seasonality": combine_keys(parts.continuation, parts.open3),
        "yday_swing_open5_seasonality": combine_keys(parts.yday_swing, parts.open5),
        "two_day_swing_open3_seasonality": combine_keys(parts.two_day_swing, parts.open3),
        "big_down_run_open3_seasonality": combine_keys(parts.down_run, parts.open3),
        "net3_swing_open3_seasonality": combine_keys(parts.net3, parts.wildness, parts.open3),
        "wildest_3d_open3_seasonality": combine_keys(parts.wildest, parts.open3),
        "vol_path_open3_seasonality": combine_keys(parts.vol_path, parts.open3),
        "vol_path_open5_seasonality": combine_keys(parts.vol_path, parts.open5),
        "yday_range_open3_seasonality": combine_keys(parts.yday_range, parts.open3),
        "vol_regime_open3_seasonality": combine_keys(parts.vol_regime, parts.open3),
        "seq3_volpath_open3_seasonality": combine_keys(parts.net3_dir, parts.vol_path, parts.open3),
        "swing_volpath_open3_seasonality": combine_keys(parts.yday_big, parts.vol_path, parts.open3),
        "streak_vol_open3_seasonality": combine_keys(parts.streak_sign, parts.vol_3d, parts.open3),
        "open_in_yday_range_seasonality": parts.loc5,
        "seq3_openloc_seasonality": combine_keys(parts.seq3, parts.loc3),
        "gap_seq3_open3_seasonality": parts.gap_seq3,
        "gap_trap_open_seasonality": parts.gap_trap,
        "streak_crash_open_seasonality": parts.streak_crash,
        "multi_crash_bounce_open_seasonality": parts.multi_crash,
    }
    return pd.DataFrame(
        {name: prior_bucket_mean(parts.session, bucket) for name, bucket in buckets.items()},
        index=history.index,
    )


def open_known_feature_row(history: pd.DataFrame, today_open: float) -> pd.Series:
    """Recompute the open-known columns as of this morning.

    `history` must be completed sessions only (through yesterday). A dummy
    today row carries `today_open` so gap / open-location buckets see this
    morning's print without inventing a close.
    """
    if history.empty or not np.isfinite(today_open):
        return pd.Series(dtype="float64")

    today_index = history.index[-1] + pd.Timedelta(1, unit="D")
    today = pd.DataFrame(
        {
            "Open": [today_open],
            "High": [today_open],
            "Low": [today_open],
            "Close": [today_open],
            "Volume": [0.0],
        },
        index=pd.DatetimeIndex([today_index]),
    )
    for column in history.columns:
        if column not in today.columns:
            today[column] = np.nan
    extended = pd.concat([history, today[history.columns]])
    return build_open_pattern_features(extended).iloc[-1]
