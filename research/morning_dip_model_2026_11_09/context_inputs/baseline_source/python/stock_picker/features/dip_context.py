"""Experimental dip measurements aligned to the morning decision date.

These rows join an already aligned training frame; passing them through
``build_training_frame`` would shift them a second time. Nothing in the default
feature pipeline or live inference calls this module. The exact initial flags
come from research/morning_dip_patterns_2026_11_09/trigger_candidates.json.

Callers must supply a complete session calendar, retaining unavailable bars as
NaN, and one compatible price/volume adjustment basis. This module cannot infer
missing exchange sessions or repair corporate actions from OHLCV alone.
Index validation checks ordering and uniqueness, not exchange-calendar
completeness. The experiment supplies SPY-calendar rows explicitly.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stock_picker.features.candle import close_location
from stock_picker.features.trend import sma
from stock_picker.features.volume import volume_zscore

LOW_WINDOW = 252
SUPPORT_WINDOW = 20
VOLATILITY_WINDOW = 20
TREND_WINDOW = 50
TREND_SLOPE_LAG = 10
# Initial research hypotheses, never learned from an outer test period.
SHOCK_DROP = 0.02
NEAR_LOW_DISTANCE = 0.02
MAX_RECOVERY_FRACTION = 0.25
MIN_CLOSE_LOCATION = 0.5
# Inclusive ratio cutoffs must not reject decimal-price equality after rounding
# (for example, 102 / 100 - 1 is slightly above 0.02 in binary floating point).
RATIO_COMPARISON_TOLERANCE = 1e-12
OHLC_TOLERANCE = 1e-5  # The frozen provider-cache validation tolerance.
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

LOW_CONTEXT_COLUMNS = (
    "dip_close_to_low_252d", "dip_open_to_low_252d", "dip_low_age_252d",
    "dip_prior_new_low_252d", "dip_open_below_low_252d", "dip_low_history_valid",
)
RECOVERY_COLUMNS = (
    "dip_loss_2d_sigma", "dip_recovery_fraction", "dip_recovery_valid",
    "dip_deep_unrecovered",
)
PRESSURE_COLUMNS = (
    "dip_loss_change_sigma", "dip_range_change_1d", "dip_volume_change_1d",
    "dip_pressure_easing", "dip_gap_previous_volume_zscore",
)
SUPPORT_AND_TREND_COLUMNS = (
    "dip_support_break_depth_20d", "dip_support_close_distance_20d",
    "dip_support_open_distance_20d", "dip_support_reclaim_20d",
    "dip_sma50_slope_10d", "dip_trend_pullback",
)
DIP_FAMILIES = {
    "low_context": LOW_CONTEXT_COLUMNS,
    "recovery": RECOVERY_COLUMNS,
    "pressure": PRESSURE_COLUMNS,
    "support_and_trend": SUPPORT_AND_TREND_COLUMNS,
}
DIP_COLUMNS = tuple(column for family in DIP_FAMILIES.values() for column in family)
DIP_FLAG_COLUMNS = (
    "dip_deep_unrecovered", "dip_pressure_easing", "dip_support_reclaim_20d",
    "dip_trend_pullback",
)
# Classification metadata only: do not add these to the default dataset path.
DIP_OPEN_KNOWN_COLUMNS = frozenset({
    "dip_open_to_low_252d", "dip_open_below_low_252d",
    "dip_recovery_fraction", "dip_recovery_valid", "dip_deep_unrecovered",
    "dip_gap_previous_volume_zscore", "dip_support_open_distance_20d",
    "dip_support_reclaim_20d",
})


def _validate_history(history: pd.DataFrame) -> None:
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("dip history requires a DatetimeIndex of session dates")
    if history.index.hasnans or history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("dip history session dates must be unique and ascending")
    if not history.index.equals(history.index.normalize()):
        raise ValueError("dip history index must contain normalized session dates")
    missing = set(PRICE_COLUMNS) - set(history.columns)
    if missing:
        raise ValueError(f"dip history is missing columns: {sorted(missing)}")


def _positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = np.isfinite(numerator) & np.isfinite(denominator) & denominator.gt(0)
    return numerator.div(denominator.where(valid)).where(valid)


def _flag(condition: pd.Series, valid: pd.Series) -> pd.Series:
    """Unknown inputs remain NaN even when pandas comparisons evaluate False."""
    return condition.astype(float).where(valid)


def _most_recent_low_age(values: np.ndarray) -> float:
    # NumPy argmin chooses the first minimum; reverse to select the latest tie.
    return float(np.argmin(values[::-1]) + 1)


def morning_dip_features(history: pd.DataFrame) -> pd.DataFrame:
    """Return candidate values for each entry date using prior bars and its Open.

    Current HLCV is allowed to be absent: the scalar helper appends exactly such
    an Open-only row. A complete prior bar requires finite positive OHLCV and a
    consistent range. Invalid bars remain on the calendar and invalidate windows
    that require them, rather than silently shortening a lookback.
    """
    _validate_history(history)
    raw = history.loc[:, list(PRICE_COLUMNS)].astype(float)
    valid = (
        np.isfinite(raw).all(axis=1)
        & raw.gt(0).all(axis=1)
        & raw.High.add(OHLC_TOLERANCE).ge(raw[["Open", "Close"]].max(axis=1))
        & raw.Low.sub(OHLC_TOLERANCE).le(raw[["Open", "Close"]].min(axis=1))
        & raw.High.ge(raw.Low)
    )
    complete = raw.where(valid, np.nan)
    open_now = raw.Open.where(np.isfinite(raw.Open) & raw.Open.gt(0))
    close, high, low, volume = (complete[name] for name in ("Close", "High", "Low", "Volume"))
    prior_close, prior_low = close.shift(1), low.shift(1)
    returns = close.pct_change(fill_method=None)
    r1, r2 = returns.shift(1), returns.shift(2)
    two_day = _positive_ratio(prior_close, close.shift(3)) - 1

    # shift(1) precedes rolling: today's eventual low never enters this floor.
    # Official pandas Rolling.apply(raw=True) passes ndarray windows to the
    # reduction. Documentation verification is recorded with the experiment.
    low_window = prior_low.rolling(LOW_WINDOW, min_periods=LOW_WINDOW)
    low252 = low_window.min()
    low_age = low_window.apply(_most_recent_low_age, raw=True)
    low_valid = valid.shift(1).rolling(LOW_WINDOW, min_periods=LOW_WINDOW).sum().eq(LOW_WINDOW)
    older_low = low.shift(2).rolling(LOW_WINDOW - 1, min_periods=LOW_WINDOW - 1).min()
    close_distance = _positive_ratio(prior_close, low252) - 1
    open_distance = _positive_ratio(open_now, low252) - 1

    # Both selloff returns are excluded from the normalization baseline.
    pre_shock_vol = returns.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std(ddof=1).shift(3)
    prior_closing_loss = close.shift(3) - prior_close
    recovery = _positive_ratio(open_now - prior_close, prior_closing_loss)
    recovery_valid = recovery.notna()
    deep_valid = low_valid & r1.notna() & r2.notna() & recovery_valid
    deep_dip = (
        r1.le(-SHOCK_DROP + RATIO_COMPARISON_TOLERANCE)
        & r2.le(-SHOCK_DROP + RATIO_COMPARISON_TOLERANCE)
        & close_distance.between(-RATIO_COMPARISON_TOLERANCE, NEAR_LOW_DISTANCE + RATIO_COMPARISON_TOLERANCE)
        & recovery.between(-RATIO_COMPARISON_TOLERANCE, MAX_RECOVERY_FRACTION + RATIO_COMPARISON_TOLERANCE)
    )

    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1, skipna=False)
    prior_location = close_location(high, low, close).shift(1)
    pressure_valid = r1.notna() & r2.notna() & volume.shift(1).notna() & volume.shift(2).notna() & np.isfinite(prior_location)
    pressure_eases = (
        r1.lt(0) & r2.lt(0) & r1.abs().lt(r2.abs())
        & volume.shift(1).lt(volume.shift(2))
        & prior_location.ge(MIN_CLOSE_LOCATION - RATIO_COMPARISON_TOLERANCE)
    )
    gap = _positive_ratio(open_now, prior_close) - 1

    # This support existed before yesterday's attempted break; excluding
    # yesterday prevents the reference level from following the break downward.
    support = low.shift(2).rolling(SUPPORT_WINDOW, min_periods=SUPPORT_WINDOW).min()
    reclaim_valid = support.notna() & prior_low.notna() & prior_close.notna() & open_now.notna()
    reclaim = prior_low.lt(support) & prior_close.ge(support) & open_now.ge(support)

    prior_sma50 = sma(close, TREND_WINDOW).shift(1)
    older_sma50 = prior_sma50.shift(TREND_SLOPE_LAG)
    trend_valid = two_day.notna() & prior_sma50.notna() & older_sma50.notna()
    trend_pullback = (
        two_day.le(-SHOCK_DROP + RATIO_COMPARISON_TOLERANCE)
        & prior_close.gt(prior_sma50) & prior_sma50.gt(older_sma50)
    )

    values = {
        "dip_close_to_low_252d": close_distance,
        "dip_open_to_low_252d": open_distance,
        "dip_low_age_252d": low_age.where(low_valid),
        "dip_prior_new_low_252d": _flag(prior_low.le(older_low), low_valid),
        "dip_open_below_low_252d": _flag(open_now.lt(low252), low_valid & open_now.notna()),
        "dip_low_history_valid": low_valid.astype(float),
        "dip_loss_2d_sigma": _positive_ratio(two_day, pre_shock_vol * np.sqrt(2)),
        "dip_recovery_fraction": recovery,
        "dip_recovery_valid": recovery_valid.astype(float),
        "dip_deep_unrecovered": _flag(deep_dip, deep_valid),
        "dip_loss_change_sigma": _positive_ratio(r1.abs() - r2.abs(), pre_shock_vol),
        "dip_range_change_1d": _positive_ratio(true_range.shift(1), true_range.shift(2)) - 1,
        "dip_volume_change_1d": _positive_ratio(volume.shift(1), volume.shift(2)) - 1,
        "dip_pressure_easing": _flag(pressure_eases, pressure_valid),
        "dip_gap_previous_volume_zscore": gap * volume_zscore(volume).shift(1),
        "dip_support_break_depth_20d": _positive_ratio(support - prior_low, support),
        "dip_support_close_distance_20d": _positive_ratio(prior_close, support) - 1,
        "dip_support_open_distance_20d": _positive_ratio(open_now, support) - 1,
        "dip_support_reclaim_20d": _flag(reclaim, reclaim_valid),
        "dip_sma50_slope_10d": _positive_ratio(prior_sma50, older_sma50) - 1,
        "dip_trend_pullback": _flag(trend_pullback, trend_valid),
    }
    return pd.DataFrame(values, index=history.index).loc[:, list(DIP_COLUMNS)].replace([np.inf, -np.inf], np.nan)


def morning_dip_row(
    prior_history: pd.DataFrame, today_open: float, as_of_date: date
) -> pd.Series:
    """Compute the same experimental row from completed history and current Open.

    This helper is deliberately not connected to production live inference.
    It fails on future/duplicate dates instead of truncating history silently.
    """
    _validate_history(prior_history)
    if not np.isfinite(today_open) or today_open <= 0:
        raise ValueError("today_open must be finite and positive")
    entry_date = pd.Timestamp(as_of_date)
    if entry_date != entry_date.normalize():
        raise ValueError("as_of_date must identify a normalized session date")
    if prior_history.index.tz is not None and entry_date.tzinfo is None:
        entry_date = entry_date.tz_localize(prior_history.index.tz)
    if str(prior_history.index.tz) != str(entry_date.tzinfo):
        raise ValueError("as_of_date timezone must match prior history")
    if len(prior_history) and prior_history.index.max() >= entry_date:
        raise ValueError("prior history must end strictly before as_of_date")
    current = pd.DataFrame(np.nan, index=pd.DatetimeIndex([entry_date]), columns=PRICE_COLUMNS)
    current.loc[entry_date, "Open"] = today_open
    combined = pd.concat([prior_history.loc[:, list(PRICE_COLUMNS)], current])
    return morning_dip_features(combined).loc[entry_date]
