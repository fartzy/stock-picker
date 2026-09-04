"""Human-readable, plain-English descriptions for every feature the pipeline
computes -- used by the dashboard/catalog_main.py, not by training.

Pattern-matched by column name so a new window on an existing feature (e.g. a future
return_7d) is covered automatically; only a genuinely new feature concept needs a new
pattern. See test_descriptions.py's completeness test, which fails if any real
feature column falls through to the unknown placeholder below.
"""

from __future__ import annotations

import re
from collections.abc import Callable

UNKNOWN_DESCRIPTION = "No description available for '{name}' -- add a pattern to descriptions.py."

_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], str]]] = []


def _day_plural(n: int) -> str:
    return "day" if n == 1 else "days"


def _pattern(regex: str, template: Callable[[re.Match], str]) -> None:
    _PATTERNS.append((re.compile(regex), template))


_pattern(
    r"^log_return_(\d+)d$",
    lambda m: f"Log return over the trailing {m[1]} trading {_day_plural(int(m[1]))}.",
)
_pattern(
    r"^return_(\d+)d$",
    lambda m: f"Percent price change over the trailing {m[1]} trading {_day_plural(int(m[1]))}.",
)
_pattern(
    r"^momentum_spread_(\d+)_(\d+)d$",
    lambda m: (
        f"{m[1]}-day return minus {m[2]}-day return -- positive means the trend is "
        f"accelerating relative to its {m[2]}-day pace, negative means it's decelerating."
    ),
)
_pattern(
    r"^consecutive_day_streak$",
    lambda m: (
        "Signed count of consecutive up/down closing days "
        "(e.g. +3 = three straight up days, -2 = two straight down days)."
    ),
)
_pattern(
    r"^volatility_(\d+)d$",
    lambda m: (
        f"Annualized standard deviation of daily returns over the trailing {m[1]} "
        "days -- a realized-volatility estimate."
    ),
)
_pattern(
    r"^atr_14$",
    lambda m: "Average True Range (14-day) -- typical daily price range, accounting for gaps.",
)
_pattern(
    r"^parkinson_vol_(\d+)d$",
    lambda m: f"Parkinson high-low range volatility estimate over the trailing {m[1]} days.",
)
_pattern(
    r"^garman_klass_vol_(\d+)d$",
    lambda m: f"Garman-Klass OHLC volatility estimate over the trailing {m[1]} days.",
)
_pattern(
    r"^price_vs_sma_(\d+)d$",
    lambda m: f"Current price relative to its {m[1]}-day simple moving average (0 = at the average).",
)
_pattern(
    r"^price_vs_ema_(\d+)d$",
    lambda m: f"Current price relative to its {m[1]}-day exponential moving average (0 = at the average).",
)
_pattern(
    r"^macd_line$",
    lambda m: "MACD line: 12-day EMA minus 26-day EMA -- a trend-following momentum indicator.",
)
_pattern(
    r"^macd_signal$",
    lambda m: "9-day EMA of the MACD line -- the signal line used to spot MACD crossovers.",
)
_pattern(
    r"^macd_hist$",
    lambda m: "MACD line minus its signal line -- shows momentum shifts before a crossover.",
)
_pattern(
    r"^rsi_(\d+)$",
    lambda m: (
        f"Relative Strength Index over {m[1]} days -- above 70 is typically "
        "overbought, below 30 oversold."
        + (" A short-term mean-reversion variant." if int(m[1]) <= 2 else "")
    ),
)
_pattern(
    r"^stochastic_k_(\d+)d$",
    lambda m: (
        f"Stochastic %K over {m[1]} days -- where today's close sits within the "
        "recent high-low range (0-100)."
    ),
)
_pattern(
    r"^stochastic_d_(\d+)d$",
    lambda m: f"3-day moving average of the {m[1]}-day stochastic %K -- the signal line for crossovers.",
)
_pattern(
    r"^bollinger_pct_b$",
    lambda m: "Where price sits within its 20-day Bollinger Bands (0 = lower band, 1 = upper band).",
)
_pattern(
    r"^bollinger_bandwidth$",
    lambda m: (
        "Width of the 20-day Bollinger Bands relative to the moving average -- "
        "widens when volatility expands."
    ),
)
_pattern(
    r"^williams_r$",
    lambda m: "Williams %R (14-day) -- similar to stochastic %K, inverted and offset; -100 to 0 scale.",
)
_pattern(
    r"^cci$",
    lambda m: "Commodity Channel Index (20-day) -- how far price has strayed from its statistical average.",
)
_pattern(
    r"^volume_ratio_(\d+)d$",
    lambda m: f"Today's volume divided by its {m[1]}-day average -- above 1 means higher-than-usual activity.",
)
_pattern(
    r"^dollar_volume$",
    lambda m: "Price times volume -- the dollar amount traded that day, a liquidity measure.",
)
_pattern(
    r"^obv$",
    lambda m: "On-Balance Volume -- a running total that adds volume on up days and subtracts it on down days.",
)
_pattern(
    r"^obv_change_(\d+)d$",
    lambda m: f"Change in On-Balance Volume over the trailing {m[1]} days -- is buying/selling pressure accelerating.",
)
_pattern(
    r"^volume_zscore_(\d+)d$",
    lambda m: f"How many standard deviations today's volume is from its {m[1]}-day average.",
)
_pattern(
    r"^overnight_gap$",
    lambda m: (
        "Percent change from yesterday's close to today's open -- the one feature "
        "legitimately known before the market opens."
    ),
)
_pattern(r"^day_range_pct$", lambda m: "Today's high-low range as a percent of the closing price.")
_pattern(r"^body_pct$", lambda m: "Percent change from today's open to today's close (the candle 'body').")
_pattern(
    r"^upper_wick_pct$",
    lambda m: "Size of the upper candle wick (high minus the higher of open/close) as a percent of the open.",
)
_pattern(
    r"^lower_wick_pct$",
    lambda m: "Size of the lower candle wick (lower of open/close minus low) as a percent of the open.",
)
_pattern(
    r"^close_location$",
    lambda m: "Where the close landed within today's high-low range (0 = at the low, 1 = at the high).",
)
_pattern(
    r"^skew_(\d+)d$",
    lambda m: f"Skewness of daily returns over the trailing {m[1]} days -- asymmetry of the return distribution.",
)
_pattern(
    r"^kurtosis_(\d+)d$",
    lambda m: f"Kurtosis of daily returns over the trailing {m[1]} days -- how fat-tailed/prone to extreme moves.",
)
_pattern(
    r"^max_drawdown_(\d+)d$",
    lambda m: f"Current drop from the highest close in the trailing {m[1]} days (always <= 0).",
)
_pattern(
    r"^return_autocorr_(\d+)d$",
    lambda m: (
        f"Correlation between consecutive daily returns over the trailing {m[1]} "
        "days -- positive means trending, negative means mean-reverting."
    ),
)
_pattern(
    r"^sharpe_(\d+)d$",
    lambda m: f"Annualized return-to-volatility ratio over the trailing {m[1]} days.",
)
_pattern(
    r"^var_5pct_(\d+)d$",
    lambda m: f"5th-percentile daily return over the trailing {m[1]} days -- a historical Value-at-Risk estimate.",
)
_pattern(r"^day_of_week$", lambda m: "Day of the week (0=Monday .. 4=Friday).")
_pattern(r"^day_of_month$", lambda m: "Calendar day of the month.")
_pattern(r"^month$", lambda m: "Calendar month (1-12).")
_pattern(
    r"^day_of_week_seasonality$",
    lambda m: "Average historical return on this specific day-of-week, using only prior occurrences.",
)
_pattern(
    r"^return_rank_(\d+)d$",
    lambda m: f"This ticker's percentile rank (0-1) for {m[1]}-day return among all tracked tickers on this date.",
)
_pattern(
    r"^beta_60d$",
    lambda m: "60-day rolling beta vs. the SPY benchmark -- sensitivity to overall market moves.",
)
_pattern(r"^correlation_60d$", lambda m: "60-day rolling correlation of daily returns vs. the SPY benchmark.")
_pattern(
    r"^relative_strength$",
    lambda m: "Cumulative return vs. the SPY benchmark since the start of the tracked history.",
)
_pattern(
    r"^sector_relative_return$",
    lambda m: (
        "Return relative to the average return of same-sector peers "
        "(not yet populated -- sector labels aren't persisted)."
    ),
)


def describe_feature(name: str) -> str:
    """Plain-English description of a feature column, or a flagged placeholder if
    no pattern matches."""
    for pattern, template in _PATTERNS:
        match = pattern.match(name)
        if match:
            return template(match)
    return UNKNOWN_DESCRIPTION.format(name=name)
