"""Concrete, worked numeric examples for every feature the pipeline computes --
"if the prior data looked like this, the feature value comes out like that."

descriptions.py explains what a feature means and formulas.py shows the actual
computation; neither answers "okay, but what does that look like in practice."
This is the third, complementary view -- short enough to scan alongside the
other two, not a paragraph. Same pattern-match-by-column-name approach as
descriptions.py/formulas.py, same fallback-on-no-match behavior, same
completeness test in test_examples.py.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from stock_picker.features.descriptions import _day_plural

UNKNOWN_EXAMPLE = "No example available for '{name}'. Add a pattern to examples.py."

_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], str]]] = []


def _pattern(regex: str, template: Callable[[re.Match], str]) -> None:
    _PATTERNS.append((re.compile(regex), template))


_pattern(
    r"^log_return_(\d+)d$",
    lambda m: f"Price up ~5% over the trailing {m[1]} {_day_plural(int(m[1]))} → about +0.049.",
)
_pattern(
    r"^return_(\d+)d$",
    lambda m: f"Price up 5% over the trailing {m[1]} {_day_plural(int(m[1]))} → +0.05.",
)
_pattern(
    r"^momentum_spread_(\d+)_(\d+)d$",
    lambda m: (
        f"Up 6% over {m[1]} {_day_plural(int(m[1]))} but only up 2% over {m[2]} {_day_plural(int(m[2]))} → +0.04 "
        "(the trend is picking up pace)."
    ),
)
_pattern(
    r"^consecutive_day_streak$",
    lambda m: "Three straight higher closes → +3. Two straight lower closes → -2.",
)
_pattern(
    r"^volatility_(\d+)d$",
    lambda m: (
        f"Daily moves have averaged about 1.5% over the trailing {m[1]} {_day_plural(int(m[1]))} "
        "→ annualizes to roughly 0.24."
    ),
)
_pattern(
    r"^atr_14$",
    lambda m: "Daily ranges (including gaps) have averaged $2.30 on a $100 stock → atr_14 ≈ 2.30.",
)
_pattern(
    r"^parkinson_vol_(\d+)d$",
    lambda m: (
        f"High-low ranges over the trailing {m[1]} {_day_plural(int(m[1]))} imply about "
        "20% annualized volatility → 0.20."
    ),
)
_pattern(
    r"^garman_klass_vol_(\d+)d$",
    lambda m: (
        f"OHLC ranges over the trailing {m[1]} {_day_plural(int(m[1]))} imply about "
        "22% annualized volatility → 0.22."
    ),
)
_pattern(
    r"^price_vs_sma_(\d+)d$",
    lambda m: f"Price is 3% above its {m[1]}-day average → +0.03.",
)
_pattern(
    r"^price_vs_ema_(\d+)d$",
    lambda m: f"Price is 3% above its {m[1]}-day exponential average → +0.03.",
)
_pattern(
    r"^macd_line$",
    lambda m: "The 12-day trend is running hotter than the 26-day trend → a positive macd_line.",
)
_pattern(
    r"^macd_signal$",
    lambda m: "macd_line has been positive and rising all week → its 9-day average (macd_signal) follows upward, a step behind.",
)
_pattern(
    r"^macd_hist$",
    lambda m: "macd_line just crossed above macd_signal → macd_hist flips from negative to positive.",
)
_pattern(
    r"^rsi_(\d+)$",
    lambda m: (
        f"Up on most of the last {m[1]} {_day_plural(int(m[1]))} with only small down days → "
        f"rsi_{m[1]} above 70 (overbought)."
    ),
)
_pattern(
    r"^stochastic_k_(\d+)d$",
    lambda m: (
        f"Today's close sits right at the top of the {m[1]}-day range → stochastic_k near 100."
    ),
)
_pattern(
    r"^stochastic_d_(\d+)d$",
    lambda m: f"stochastic_k_{m[1]}d spiked to 95 then eased to 88 then 90 → stochastic_d smooths that to ~91.",
)
_pattern(
    r"^bollinger_pct_b$",
    lambda m: "Price sitting right at the upper Bollinger Band → bollinger_pct_b ≈ 1.0.",
)
_pattern(
    r"^bollinger_bandwidth$",
    lambda m: "Daily swings have picked up this week → the bands widen and bollinger_bandwidth rises.",
)
_pattern(
    r"^williams_r$",
    lambda m: "Close near the top of its 14-day range → williams_r near 0 (overbought, same read as stochastic).",
)
_pattern(
    r"^cci$",
    lambda m: "Price well above its 20-day statistical average → a high positive cci.",
)
_pattern(
    r"^volume_ratio_(\d+)d$",
    lambda m: f"Today's volume is double the {m[1]}-day average → 2.0.",
)
_pattern(
    r"^dollar_volume$",
    lambda m: "1M shares traded at $50 → dollar_volume = $50,000,000.",
)
_pattern(
    r"^obv$",
    lambda m: "Up day adds today's volume, down day subtracts it → a rising obv means buying volume has dominated.",
)
_pattern(
    r"^obv_change_(\d+)d$",
    lambda m: f"obv has climbed steadily over the trailing {m[1]} {_day_plural(int(m[1]))} → a positive obv_change_{m[1]}d.",
)
_pattern(
    r"^volume_zscore_(\d+)d$",
    lambda m: f"Today's volume is 3 standard deviations above its {m[1]}-day average → volume_zscore ≈ 3.0.",
)
_pattern(
    r"^overnight_gap$",
    lambda m: "Closed at $100 yesterday, opens at $102 today → overnight_gap = +0.02.",
)
_pattern(
    r"^gap_volume_interaction$",
    lambda m: "A +2% gap on volume 3 standard deviations above normal → +0.02 * 3.0 = +0.06.",
)
_pattern(
    r"^gap_streak$",
    lambda m: "Three straight gap-up mornings → +3. Two straight gap-down mornings → -2.",
)
_pattern(r"^day_range_pct$", lambda m: "High $102, low $98, close $100 → day_range_pct = 0.04.")
_pattern(r"^body_pct$", lambda m: "Opened at $100, closed at $103 → body_pct = +0.03.")
_pattern(
    r"^upper_wick_pct$",
    lambda m: "High $105 but open/close both near $100 → a long upper wick, upper_wick_pct ≈ 0.05.",
)
_pattern(
    r"^lower_wick_pct$",
    lambda m: "Low $95 but open/close both near $100 → a long lower wick, lower_wick_pct ≈ 0.05.",
)
_pattern(
    r"^close_location$",
    lambda m: "Close landed at today's high → close_location = 1.0. Close at the low → 0.0.",
)
_pattern(
    r"^skew_(\d+)d$",
    lambda m: f"Mostly small gains punctuated by one sharp drop over the trailing {m[1]} {_day_plural(int(m[1]))} → negative skew.",
)
_pattern(
    r"^kurtosis_(\d+)d$",
    lambda m: f"Calm returns except for a couple of extreme days in the trailing {m[1]} {_day_plural(int(m[1]))} → high kurtosis (fat tails).",
)
_pattern(
    r"^max_drawdown_(\d+)d$",
    lambda m: f"Peaked at $110, now at $99 within the trailing {m[1]} {_day_plural(int(m[1]))} → max_drawdown ≈ -0.10.",
)
_pattern(
    r"^return_autocorr_(\d+)d$",
    lambda m: f"Up days have tended to be followed by more up days over the trailing {m[1]} {_day_plural(int(m[1]))} → positive autocorrelation.",
)
_pattern(
    r"^sharpe_(\d+)d$",
    lambda m: f"Steady positive returns with low volatility over the trailing {m[1]} {_day_plural(int(m[1]))} → a high sharpe_{m[1]}d.",
)
_pattern(
    r"^var_5pct_(\d+)d$",
    lambda m: f"The worst ~5% of daily returns over the trailing {m[1]} {_day_plural(int(m[1]))} have been around -3% → var_5pct_{m[1]}d ≈ -0.03.",
)
_pattern(r"^day_of_week$", lambda m: "A Wednesday → day_of_week = 2 (0=Monday).")
_pattern(r"^day_of_month$", lambda m: "The 15th of the month → day_of_month = 15.")
_pattern(r"^month$", lambda m: "March → month = 3.")
_pattern(
    r"^day_of_week_seasonality$",
    lambda m: "This ticker has averaged +0.3% on prior Mondays → day_of_week_seasonality ≈ 0.003 on a Monday.",
)
_pattern(
    r"^setup_seasonality$",
    lambda m: (
        "The last 5 times this ticker gapped down ~1% after a down day, it averaged "
        "+0.4% that same day → setup_seasonality ≈ 0.004."
    ),
)
_pattern(
    r"^pooled_setup_seasonality$",
    lambda m: (
        "Same idea as setup_seasonality, averaged across every tracked ticker's prior "
        "occurrences of that setup instead of just this one's."
    ),
)
_pattern(
    r"^day_session_streak$",
    lambda m: "Two straight day-session (open-to-close) gains → day_session_streak = +2.",
)
_pattern(
    r"^day_session_streak_seasonality$",
    lambda m: (
        "The last 10 times this ticker was on a +2 day-session streak, it averaged "
        "+0.2% the next day → ≈ 0.002."
    ),
)
_pattern(
    r"^pattern_sequence_seasonality_3d$",
    lambda m: (
        "The last 10 times this ticker went down-down-up over 3 days, it averaged "
        "+0.5% next → ≈ 0.005."
    ),
)
_pattern(
    r"^weekday_lag_return$",
    lambda m: "Up 1.5% last Tuesday → weekday_lag_return = 0.015 on this Tuesday.",
)
_pattern(
    r"^weekday_lag_seasonality$",
    lambda m: (
        "The last 10 Tuesdays that followed a similar move the Tuesday before "
        "averaged +0.3% → ≈ 0.003."
    ),
)
_pattern(
    r"^return_rank_(\d+)d$",
    lambda m: f"This ticker's {m[1]}-day return beats 80% of the tracked universe today → return_rank_{m[1]}d = 0.80.",
)
_pattern(
    r"^beta_60d$",
    lambda m: "This ticker moves about 1.2x as much as SPY on average → beta_60d = 1.2.",
)
_pattern(
    r"^correlation_60d$",
    lambda m: "This ticker's daily moves have tracked SPY closely over 60 days → correlation_60d ≈ 0.85.",
)
_pattern(
    r"^relative_strength$",
    lambda m: "Up 40% since tracking started while SPY is up 10% → relative_strength ≈ +0.30 ahead of the benchmark.",
)
_pattern(
    r"^sector_relative_return$",
    lambda m: "Not yet populated -- sector labels aren't persisted, so there's no peer group to compare against yet.",
)


def feature_example(name: str) -> str:
    """A short worked example for a feature column, or a flagged placeholder if
    no pattern matches."""
    for pattern, template in _PATTERNS:
        match = pattern.match(name)
        if match:
            return template(match)
    return UNKNOWN_EXAMPLE.format(name=name)
