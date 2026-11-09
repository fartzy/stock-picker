"""Exploratory inference for the repaired Rank 1, down-day next-open scenario.

Fixed saved selections, $10,000 per position; no model reranking. The tests
do not adjust for choosing the strategy after looking at these results.
The t-test assumes independent observations; bootstrap diagnostics are
sensitivity checks, not proof of execution feasibility or a future edge.
"""
from collections import Counter
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
ROWS = json.loads((ROOT / "rank1_significance_rows.json").read_text())
REPETITIONS = 200_000
SEED = 20261106


def summarize(values):
    test = stats.ttest_1samp(values, popmean=0)
    interval = test.confidence_interval(confidence_level=.95)
    directional = stats.ttest_1samp(values, popmean=0, alternative="greater")
    return {
        "n": len(values), "pnl": float(values.sum()), "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)), "median": float(np.median(values)),
        "wins": int((values > 0).sum()), "losses": int((values < 0).sum()),
        "mean_win": float(values[values > 0].mean()), "mean_loss": float(values[values < 0].mean()),
        "t_statistic": float(test.statistic), "df": float(test.df),
        "p_two_sided": float(test.pvalue), "p_positive_mean_one_sided": float(directional.pvalue),
        "mean_95pct_t_interval": [float(interval.low), float(interval.high)],
    }


x = np.array([row["pnl"] for row in ROWS])
assert len(x) == 36 and round(float(x.sum()), 2) == 2708.87
rng = np.random.default_rng(SEED)
indices = rng.integers(0, len(x), size=(REPETITIONS, len(x)))
bootstrap_means = x[indices].mean(axis=1)
observed_mean = x.mean()
null_means = bootstrap_means - observed_mean

week_labels = [
    (date.fromisoformat(row["date"]) - timedelta(days=date.fromisoformat(row["date"]).weekday())).isoformat()
    for row in ROWS
]
weeks = sorted(set(week_labels))
week_totals = np.array([sum(row["pnl"] for row, week in zip(ROWS, week_labels) if week == label) for label in weeks])
week_sizes = np.array([week_labels.count(label) for label in weeks])
week_indices = rng.integers(0, len(weeks), size=(REPETITIONS, len(weeks)))
week_means = week_totals[week_indices].sum(axis=1) / week_sizes[week_indices].sum(axis=1)

largest = sorted(ROWS, key=lambda row: row["pnl"], reverse=True)[:3]
smallest = sorted(ROWS, key=lambda row: row["pnl"])[:3]
win_test = stats.binomtest(int((x > 0).sum()), n=len(x), p=.5, alternative="greater")
result = {
    "scenario": "Repaired Rank 1, $10K, sell winners at close and losers at next open",
    "population": {"first_date": ROWS[0]["date"], "last_date": ROWS[-1]["date"],
                   "positions": len(ROWS), "calendar_weeks": len(weeks), "tickers": dict(Counter(row["ticker"] for row in ROWS))},
    "gross": summarize(x),
    "assumed_11bps_each_way": summarize(np.array([row["net11"] for row in ROWS])),
    "assumed_26bps_each_way": summarize(np.array([row["net26"] for row in ROWS])),
    "bootstrap": {
        "repetitions": REPETITIONS, "seed": SEED,
        "iid_percentile_mean_95pct_interval": np.quantile(bootstrap_means, [.025, .975]).tolist(),
        "centered_iid_null_one_sided_p": float((1 + (null_means >= observed_mean).sum()) / (REPETITIONS + 1)),
        "calendar_week_cluster_mean_95pct_interval": np.quantile(week_means, [.025, .975]).tolist(),
        "week_note": "Resample whole weeks with replacement; eight clusters are too few for precise inference.",
    },
    "win_frequency_only": {"null_win_probability": .5, "p_one_sided": float(win_test.pvalue),
                           "note": "Tests win frequency, not profitable expectancy; ignores unequal win/loss sizes."},
    "tail_sensitivity": {"top_three": largest, "bottom_three": smallest,
                         "top_three_pnl": sum(row["pnl"] for row in largest),
                         "remaining_33_pnl": float(x.sum() - sum(row["pnl"] for row in largest)),
                         "note": "Diagnostic only; deleting winners is not a proposed trading strategy."},
    "limits": [
        "Exploratory result: Rank size and exit alternatives have already been examined on these dates.",
        "Opening-price entries occur before the morning selections are published; actual fills are unverified.",
        "A zero-mean test is not a risk-adjusted alpha test or a comparison against random stocks.",
        "Assumed trading costs are not measured execution costs; financing and taxes are excluded.",
        "Short sample, repeated names and shared market conditions can invalidate independence assumptions.",
        "A p-value is not the probability that the strategy has no edge or that the result was just luck.",
    ],
}
(ROOT / "rank1_significance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
