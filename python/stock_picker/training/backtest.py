"""Simulates the long-only, threshold-gated day-session strategy: buy at the open on
any day the model's predicted return exceeds `threshold`, sell at the close, sit out
otherwise.

No "pick the best threshold" function on purpose: with a handful of holdout tickers over
a few months, any single threshold a sweep highlights is easily a small-sample fluke.
`sweep_thresholds` reports `n_trades` alongside every metric so the tradeoff and the
sample size stay visible together -- read this as illustrating the mechanism, not as a
trading recommendation, until the universe and history are much larger.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03]


def simulate_trades(predicted: pd.Series, actual: pd.Series, threshold: float, n_days: int | None = None) -> dict:
    """P&L if we'd bought whenever predicted > threshold, sold at the close same day.

    `n_days` (the number of distinct trading days the sweep covers, across
    however many tickers), when given, adds `avg_picks_per_day` -- `n_trades`
    alone is a total across the whole evaluation window (e.g. a year of
    holdout tickers), which answers "how many trades total" but not "how many
    stocks would this actually flag on any given day," the more intuitive
    question for someone deciding whether to check the screen today.
    """
    taken = predicted > threshold
    n_trades = int(taken.sum())
    trade_returns = actual[taken]
    has_trades = len(trade_returns) > 0

    return {
        "threshold": threshold,
        "n_trades": n_trades,
        "avg_picks_per_day": (n_trades / n_days) if n_days else None,
        "hit_rate": float((trade_returns > 0).mean()) if has_trades else float("nan"),
        "total_return": float(trade_returns.sum()) if has_trades else float("nan"),
        "avg_return": float(trade_returns.mean()) if has_trades else float("nan"),
        "return_std": float(trade_returns.std()) if len(trade_returns) > 1 else float("nan"),
    }


def sweep_thresholds(
    predicted: pd.Series,
    actual: pd.Series,
    thresholds: list[float] = DEFAULT_THRESHOLDS,
    n_days: int | None = None,
) -> pd.DataFrame:
    return pd.DataFrame([simulate_trades(predicted, actual, t, n_days=n_days) for t in thresholds])


def rank_ic(predicted: pd.Series, actual: pd.Series, dates: pd.Series) -> float:
    """Mean daily cross-sectional Spearman rank correlation (Rank IC) between
    predicted and actual returns -- the standard quant metric for "does this
    model rank tickers well against each other on a given day," a genuinely
    different question from directional_accuracy's per-row sign check, which
    has no notion of "relative to that day's other tickers." Averaging each
    day's own correlation, rather than one correlation over every pooled row,
    is what makes this a cross-sectional metric instead of a pooled one --
    the exact way `predicted`/`actual`/`dates` mix ticker-days together
    otherwise.

    A day with only one ticker (correlation undefined) contributes NaN, which
    pandas' mean() silently skips -- expected for the sparsely-covered
    beginning of a held-out-ticker window, not a bug.
    """
    frame = pd.DataFrame({"predicted": predicted, "actual": actual, "date": dates})

    def _daily_ic(day: pd.DataFrame) -> float:
        if len(day) < 2:
            return float("nan")
        return day["predicted"].corr(day["actual"], method="spearman")

    daily_ic = frame.groupby("date").apply(_daily_ic, include_groups=False)
    return float(daily_ic.mean())
