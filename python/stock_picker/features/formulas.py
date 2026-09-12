"""Actual computation for every feature the pipeline computes, as a short pandas
expression -- shown alongside descriptions.py's prose so the real formula, not just
a restatement of it, is one click away. Mirrors descriptions.py exactly: same
pattern-match-by-column-name approach, same fallback-on-no-match behavior. See
test_formulas.py's completeness test, which fails if any real feature column falls
through to the unknown placeholder below.
"""

from __future__ import annotations

import re
from collections.abc import Callable

UNKNOWN_FORMULA = "No formula available for '{name}'. Add a pattern to formulas.py."

_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], str]]] = []


def _pattern(regex: str, template: Callable[[re.Match], str]) -> None:
    _PATTERNS.append((re.compile(regex), template))


_pattern(r"^log_return_(\d+)d$", lambda m: f"log(close / close.shift({m[1]}))")
_pattern(r"^return_(\d+)d$", lambda m: f"close.pct_change({m[1]})")
_pattern(
    r"^momentum_spread_(\d+)_(\d+)d$",
    lambda m: f"close.pct_change({m[1]}) - close.pct_change({m[2]})",
)
_pattern(
    r"^consecutive_day_streak$",
    lambda m: "running signed count of consecutive up/down closes; resets to 0 on a flat day",
)
_pattern(
    r"^volatility_(\d+)d$",
    lambda m: f"close.pct_change().rolling({m[1]}).std() * sqrt(252)",
)
_pattern(r"^volatility_delta_(\d+)d$", lambda m: f"volatility_20d.diff({m[1]})")
_pattern(
    r"^atr_14$",
    lambda m: "max(high-low, |high-prior_close|, |low-prior_close|).rolling(14).mean()",
)
_pattern(
    r"^parkinson_vol_(\d+)d$",
    lambda m: f"sqrt((log(high/low)**2).rolling({m[1]}).mean() / (4*log(2))) * sqrt(252)",
)
_pattern(
    r"^garman_klass_vol_(\d+)d$",
    lambda m: (
        f"sqrt((0.5*log(high/low)**2 - (2*log(2)-1)*log(close/open)**2)"
        f".rolling({m[1]}).mean()) * sqrt(252)"
    ),
)
_pattern(r"^price_vs_sma_(\d+)d$", lambda m: f"close / close.rolling({m[1]}).mean() - 1")
_pattern(r"^price_vs_ema_(\d+)d$", lambda m: f"close / close.ewm(span={m[1]}).mean() - 1")
_pattern(r"^macd_line$", lambda m: "ema(close, 12) - ema(close, 26)")
_pattern(r"^macd_signal$", lambda m: "macd_line.ewm(span=9).mean()")
_pattern(r"^macd_hist$", lambda m: "macd_line - macd_signal")
_pattern(
    r"^rsi_(\d+)$",
    lambda m: (
        f"gain = close.diff().clip(lower=0); loss = -close.diff().clip(upper=0); "
        f"100 - 100/(1 + gain.ewm(alpha=1/{m[1]}).mean() / loss.ewm(alpha=1/{m[1]}).mean())"
    ),
)
_pattern(
    r"^stochastic_k_(\d+)d$",
    lambda m: f"100 * (close - low.rolling({m[1]}).min()) / (high.rolling({m[1]}).max() - low.rolling({m[1]}).min())",
)
_pattern(r"^stochastic_d_(\d+)d$", lambda m: f"stochastic_k_{m[1]}d.rolling(3).mean()")
_pattern(
    r"^bollinger_pct_b$",
    lambda m: "(close - (sma20 - 2*std20)) / ((sma20 + 2*std20) - (sma20 - 2*std20))",
)
_pattern(r"^bollinger_bandwidth$", lambda m: "(4 * std20) / sma20")
_pattern(
    r"^williams_r$",
    lambda m: "-100 * (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min())",
)
_pattern(
    r"^cci$",
    lambda m: (
        "typical = (high+low+close)/3; "
        "(typical - typical.rolling(20).mean()) / (0.015 * mean_abs_deviation(typical, 20))"
    ),
)
_pattern(r"^volume_ratio_(\d+)d$", lambda m: f"volume / volume.rolling({m[1]}).mean()")
_pattern(r"^dollar_volume$", lambda m: "close * volume")
_pattern(r"^obv$", lambda m: "(sign(close.diff()) * volume).cumsum()")
_pattern(r"^obv_change_(\d+)d$", lambda m: f"obv.diff({m[1]})")
_pattern(
    r"^volume_zscore_(\d+)d$",
    lambda m: f"(volume - volume.rolling({m[1]}).mean()) / volume.rolling({m[1]}).std()",
)
_pattern(r"^overnight_gap$", lambda m: "(open - close.shift(1)) / close.shift(1)")
_pattern(
    r"^gap_volume_interaction$",
    lambda m: "overnight_gap * ((volume - volume.rolling(20).mean()) / volume.rolling(20).std())",
)
_pattern(r"^gap_streak$", lambda m: "signed running count of consecutive same-direction overnight gaps")
_pattern(r"^day_range_pct$", lambda m: "(high - low) / close")
_pattern(r"^body_pct$", lambda m: "(close - open) / open")
_pattern(r"^upper_wick_pct$", lambda m: "(high - max(open, close)) / open")
_pattern(r"^lower_wick_pct$", lambda m: "(min(open, close) - low) / open")
_pattern(r"^close_location$", lambda m: "(close - low) / (high - low)")
_pattern(r"^skew_(\d+)d$", lambda m: f"close.pct_change().rolling({m[1]}).skew()")
_pattern(r"^kurtosis_(\d+)d$", lambda m: f"close.pct_change().rolling({m[1]}).kurt()")
_pattern(
    r"^max_drawdown_(\d+)d$",
    lambda m: f"close / close.rolling({m[1]}, min_periods=1).max() - 1",
)
_pattern(
    r"^return_autocorr_(\d+)d$",
    lambda m: f"returns.rolling({m[1]}).corr(returns.shift(1))",
)
_pattern(
    r"^sharpe_(\d+)d$",
    lambda m: f"(returns.rolling({m[1]}).mean() / returns.rolling({m[1]}).std()) * sqrt(252)",
)
_pattern(r"^var_5pct_(\d+)d$", lambda m: f"returns.rolling({m[1]}).quantile(0.05)")
_pattern(r"^day_of_week$", lambda m: "date.dayofweek")
_pattern(r"^day_of_month$", lambda m: "date.day")
_pattern(r"^month$", lambda m: "date.month")
_pattern(
    r"^day_of_week_seasonality$",
    lambda m: "close.pct_change().groupby(date.dayofweek).transform(lambda s: s.expanding().mean())",
)
_pattern(
    r"^setup_seasonality$",
    lambda m: (
        "bucket = bucket3(gap) + '_' + bucket3(daily_return.shift(1)); "
        "daily_return.groupby(bucket).transform(lambda s: s.expanding().mean())"
    ),
)
_pattern(
    r"^pooled_setup_seasonality$",
    lambda m: (
        "per (date, bucket): pooled_returns.groupby(['date', 'bucket']).sum()/count(); "
        "cumsum per bucket, shift(1) by date so date t only sees dates < t"
    ),
)
_pattern(
    r"^day_session_streak$",
    lambda m: (
        "running signed, capped(+/-3) count of consecutive up/down "
        "(close-open)/open days; resets to 0 on NaN"
    ),
)
_pattern(
    r"^day_session_streak_seasonality$",
    lambda m: (
        "day_session_return.groupby(day_session_streak)"
        ".transform(lambda s: s.rolling(10, min_periods=1).mean())"
    ),
)
_pattern(
    r"^pattern_sequence_seasonality_3d$",
    lambda m: (
        "day_session_return.groupby(sign(day_session_return).rolling(3)->'UDU' key)"
        ".transform(lambda s: s.rolling(10, min_periods=1).mean())"
    ),
)
_pattern(r"^weekday_lag_return$", lambda m: "day_session_return.shift(5)")
_pattern(
    r"^weekday_lag_seasonality$",
    lambda m: (
        "day_session_return.groupby(date.dayofweek + '_' + bucket3(weekday_lag_return))"
        ".transform(lambda s: s.rolling(10, min_periods=1).mean())"
    ),
)
_pattern(
    r"^return_rank_(\d+)d$",
    lambda m: f"universe_returns_{m[1]}d.rank(axis=1, pct=True)[this_ticker]",
)
_pattern(
    r"^beta_60d$",
    lambda m: "returns.rolling(60).cov(spy_returns) / spy_returns.rolling(60).var()",
)
_pattern(r"^correlation_60d$", lambda m: "returns.rolling(60).corr(spy_returns)")
_pattern(
    r"^relative_strength$",
    lambda m: "(close/close.iloc[0] - 1) - (spy_close/spy_close.iloc[0] - 1)",
)
_pattern(
    r"^sector_relative_return$",
    lambda m: "daily_return - sector_avg_return",
)
_pattern(r"^spy_overnight_gap$", lambda m: "(spy_open - spy_close.shift(1)) / spy_close.shift(1)")
_pattern(r"^vix_close$", lambda m: "vix['Close']")
_pattern(
    r"^seq(\d+)_open(\d+)_seasonality$",
    lambda m: (
        f"bucket = U/D path of session.shift(1..{m[1]}) + gap_bucket{m[2]}(open); "
        "session.groupby(bucket).transform(lambda s: s.expanding().mean().shift(1))"
    ),
)
_pattern(
    r"^streak_open(\d+)_seasonality$",
    lambda m: (
        f"bucket = streak.shift(1) + gap_bucket{m[1]}(open); "
        "session.groupby(bucket).transform(lambda s: s.expanding().mean().shift(1))"
    ),
)
_pattern(
    r"^flips_(\d+)d_open3_seasonality$",
    lambda m: (
        f"bucket = flip_count(session.shift(1..{m[1]})) + last_dir + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^late_reversal_open3_seasonality$",
    lambda m: (
        "bucket = (yday opposite 3d majority?) + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^continuation_open3_seasonality$",
    lambda m: (
        "bucket = (last 3 all same dir?) + last_dir + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^yday_swing_open5_seasonality$",
    lambda m: (
        "bucket = size/dir of session.shift(1) vs vol_20d + gap5(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^two_day_swing_open3_seasonality$",
    lambda m: (
        "bucket = large?(session.shift(2)) + large?(session.shift(1)) + last_dir + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^big_down_run_open3_seasonality$",
    lambda m: (
        "bucket = consecutive large-down completed days (0/1/2+) + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^net3_swing_open3_seasonality$",
    lambda m: (
        "bucket = bucket3(sum(session.shift(1..3))) + quiet/normal/wild + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^wildest_3d_open3_seasonality$",
    lambda m: (
        "bucket = max(|session.shift(1..3)|)/vol_20d as quiet/normal/wild + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^vol_path_open(\d+)_seasonality$",
    lambda m: (
        f"bucket = (vol_20d.shift(1)-vol_20d.shift(4) sign) + (yday vol delta sign) "
        f"+ gap{m[1]}(open); session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^yday_range_open3_seasonality$",
    lambda m: (
        "bucket = (high-low).shift(1)/atr_14 as narrow/normal/wide + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^vol_regime_open3_seasonality$",
    lambda m: (
        "bucket = vol_20d vs 60d median as low/mid/high + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^seq3_volpath_open3_seasonality$",
    lambda m: (
        "bucket = sign(sum(session.shift(1..3))) + vol_path + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^swing_volpath_open3_seasonality$",
    lambda m: (
        "bucket = (|session.shift(1)| > vol_20d?) + vol_path + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^streak_vol_open3_seasonality$",
    lambda m: (
        "bucket = sign(streak.shift(1)) + 3d vol expanding/contracting + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^open_in_yday_range_seasonality$",
    lambda m: (
        "bucket = open vs yesterday high/low (below/low/mid/high/above); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^seq3_openloc_seasonality$",
    lambda m: (
        "bucket = U/D path of session.shift(1..3) + open in yday range (low/mid/high); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^gap_seq3_open3_seasonality$",
    lambda m: (
        "bucket = gap3(gap.shift(2)) + gap3(gap.shift(1)) + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^gap_trap_open_seasonality$",
    lambda m: (
        "bucket = sign(gap.shift(1)) + sign(session.shift(1)) + gap3(open); "
        "session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^streak_crash_open_seasonality$",
    lambda m: (
        "bucket = streak.shift(2) as up3/down3/other + (session.shift(1) < -2 ATR) "
        "+ gap3(open); session.groupby(bucket).expanding().mean().shift(1)"
    ),
)
_pattern(
    r"^multi_crash_bounce_open_seasonality$",
    lambda m: (
        "bucket = (last 4 sessions all down?) + (|session.shift(1)| > 2.5 ATR) "
        "+ gap5(open); session.groupby(bucket).expanding().mean().shift(1)"
    ),
)


def describe_computation(name: str) -> str:
    """Short pandas-expression formula for a feature column, or a flagged
    placeholder if no pattern matches."""
    for pattern, template in _PATTERNS:
        match = pattern.match(name)
        if match:
            return template(match)
    return UNKNOWN_FORMULA.format(name=name)
