"""Exploratory two-week comparisons using the newly specified $1,500 budget."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import gzip
import json
import sys

import numpy as np
import pandas as pd
from minute_data import regular_frame, stability_flags
from simulation import Policy, simulate, summarize, clustered_mean_ci

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'two_week_config.json').read_text())
CONDITIONAL = '--conditional-overnight' in sys.argv
PREFIX = 'conditional_overnight' if CONDITIONAL else 'two_week'
POLICIES = [
    Policy('single_2500_target8', loss_limit=1200, target=.08),
    Policy('upfront_10000_target8', max_tranches=4, initial_tranches=4, loss_limit=1200, target=.08),
    Policy('ladder5_time', max_tranches=4, loss_limit=1200),
    Policy('ladder5_target4', max_tranches=4, loss_limit=1200, target=.04),
    Policy('ladder5_target8', max_tranches=4, loss_limit=1200, target=.08),
    Policy('ladder5_target12', max_tranches=4, loss_limit=1200, target=.12),
    Policy('ladder7p5_target8', max_tranches=4, step=.075, loss_limit=1200, target=.08),
    Policy('ladder5_stable_target8', max_tranches=4, loss_limit=1200, target=.08, stability=True),
]
if CONDITIONAL:
    POLICIES = [
        Policy('single2500_target2_gate3pct', loss_limit=1200, target=.02, overnight_loss_limit=600, overnight_loss_fraction=.03),
        Policy('upfront10000_target2_gate3pct', max_tranches=4, initial_tranches=4, loss_limit=1200, target=.02, overnight_loss_limit=600, overnight_loss_fraction=.03),
        Policy('ladder5_target2_no_gate', max_tranches=4, loss_limit=1200, target=.02),
        Policy('ladder5_target2_gate600', max_tranches=4, loss_limit=1200, target=.02, overnight_loss_limit=600),
        Policy('ladder5_target2_gate3pct', max_tranches=4, loss_limit=1200, target=.02, overnight_loss_limit=600, overnight_loss_fraction=.03),
        Policy('ladder5_target4_gate3pct', max_tranches=4, loss_limit=1200, target=.04, overnight_loss_limit=600, overnight_loss_fraction=.03),
    ]
    CONFIG['holding_intent'] = 'Exit on the first profit, risk, overnight-eligibility or time condition; two weeks is only the outer limit.'
    CONFIG['default_policy'] = 'ladder5_target2_gate3pct'
    CONFIG['overnight_review'] = 'Completed minute close near 15:55 ET (15:50–15:58 if needed), with a same-session next-open execution available within two minutes.'
    CONFIG['overnight_gate'] = 'Marked whole-idea loss must be no worse than the smaller of $600 and 3% of total dollars purchased.'
    CONFIG['target_return_candidates'] = [.02, .04]
CONFIG['policies'] = [asdict(p) for p in POLICIES]
(ROOT / f'{PREFIX}_config.json').write_text(json.dumps(CONFIG, indent=2))
events = json.loads((ROOT / 'two_week_sample.json').read_text())
daily = pd.read_pickle(ROOT / 'five_year_arrays.pkl')
results, coverage, excluded = [], [], []


def load(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


for n, event in enumerate(events):
    ticker = event['ticker']
    original = ROOT / 'source_data/minute_sample_five_year' / f"{ticker}_{event['entry_date']}_{event['end_date_5']}.json.gz"
    extension = ROOT / 'source_data/minute_two_week_extension' / f"{ticker}_{event['extension_start']}_{event['end_date_10']}.json.gz"
    if not extension.exists():
        excluded.append({**event, 'reason': 'missing_extension'}); continue
    bars = load(original)['bars'] + load(extension)['bars']
    frame = regular_frame({'bars': bars})
    # Exit at the last session close before the 14-day anniversary. Including
    # that date's closing bar would exceed two weeks from the entry-time open.
    deadline = pd.Timestamp(event['entry_date']) + pd.Timedelta(days=CONFIG['maximum_calendar_days'])
    expected_dates = [day for day in event['expected_dates'] if pd.Timestamp(day) < deadline]
    frame = frame[frame.index.date < deadline.date()]
    if frame.empty or frame.index.has_duplicates:
        excluded.append({**event, 'reason': 'empty_or_duplicate_times'}); continue
    groups = {str(day): group for day, group in frame.groupby(frame.index.date)}
    if sorted(groups) != expected_dates:
        excluded.append({**event, 'reason': 'missing_session'}); continue
    invalid_coverage = False
    basis_errors = []
    reference = daily[ticker]['bars'][event['entry_idx']:event['entry_idx'] + len(groups)]
    for i, day in enumerate(expected_dates):
        group = groups[day]
        minutes = group.index.hour * 60 + group.index.minute
        if len(group) < 180 or minutes.min() > 575 or minutes.max() < 955:
            invalid_coverage = True
        basis_errors.extend([abs(group.iloc[0].o / reference[i, 0] - 1), abs(group.iloc[-1].c / reference[i, 3] - 1)])
    if invalid_coverage or max(basis_errors) > .02:
        excluded.append({**event, 'reason': 'coverage' if invalid_coverage else 'daily_minute_basis_mismatch'}); continue
    a = frame[['o', 'h', 'l', 'c', 'v']].to_numpy(dtype=float)
    if not (np.isfinite(a).all() and (a[:, :4] > 0).all() and
            (a[:, 1] + 1e-5 >= np.maximum(a[:, 0], a[:, 3])).all() and
            (a[:, 2] - 1e-5 <= np.minimum(a[:, 0], a[:, 3])).all()):
        excluded.append({**event, 'reason': 'invalid_ohlcv'}); continue
    overnight_check = np.zeros(len(frame), dtype=bool)
    if CONDITIONAL:
        slot_failure = False
        for day in expected_dates:
            slots = []
            for i in np.flatnonzero(frame.index.date == pd.Timestamp(day).date()):
                minute = frame.index[i].hour * 60 + frame.index[i].minute
                if (950 <= minute <= 958 and i + 1 < len(frame) and
                    frame.index[i + 1].date() == frame.index[i].date() and
                    (frame.index[i + 1] - frame.index[i]).total_seconds() <= 120):
                    slots.append((abs(minute - 955), i))
            if not slots:
                slot_failure = True
                break
            first_review = min(slots)[1]
            for _, i in slots:
                if i >= first_review:
                    overnight_check[i] = True
        if slot_failure:
            excluded.append({**event, 'reason': 'no_executable_close_review'}); continue
    coverage.append({'ticker': ticker, 'entry_date': event['entry_date'], 'end_date': expected_dates[-1],
                     'split': event['split'], 'sessions': len(groups), 'bars': len(frame), 'max_basis_error': max(basis_errors)})
    times = frame.index.as_unit('s').asi8.astype(float)
    stable = stability_flags(frame)
    for cost in CONFIG['cost_per_side_bps']:
        for policy in POLICIES:
            result = simulate(a, times, stable, policy, tranche=CONFIG['cash_per_purchase'], cost=cost / 10000,
                              overnight_check=overnight_check if CONDITIONAL else None)
            results.append({'ticker': ticker, 'entry_date': event['entry_date'], 'split': event['split'],
                            'policy': policy.name, 'cost_bps': cost, **result})
    if (n + 1) % 40 == 0:
        print('Examined', n + 1, 'retained', len(coverage), flush=True)

frame = pd.DataFrame(results)
frame.to_parquet(ROOT / f'{PREFIX}_results.parquet', index=False)
summary = []
for (split, cost, policy), group in frame.groupby(['split', 'cost_bps', 'policy']):
    summary.append({'split': split, 'cost_bps': int(cost), 'policy': policy, **summarize(group),
                    'loss_over_1500_pct': float((group.pnl < -1500).mean() * 100),
                    'mark_below_minus1500_pct': float((group.worst_mark < -1500).mean() * 100),
                    'avg_duration_days': float(group.duration_days.mean()),
                    'target_exit_pct': float(group.exit_reason.eq('target').mean() * 100),
                    'time_exit_pct': float(group.exit_reason.eq('time').mean() * 100)})
    summary[-1]['overnight_gate_exit_pct'] = float(group.exit_reason.eq('overnight').mean() * 100)
    summary[-1]['same_session_exit_pct'] = float((group.duration_days < 1).mean() * 100)
(ROOT / f'{PREFIX}_summary.json').write_text(json.dumps(summary, indent=2))
(ROOT / f'{PREFIX}_coverage.json').write_text(json.dumps({'input_events': len(events), 'retained': len(coverage),
    'split_counts': dict(Counter(e['split'] for e in coverage)), 'excluded_counts': dict(Counter(e['reason'] for e in excluded)),
    'coverage': coverage, 'excluded': excluded}, indent=2))
later = frame[(frame.split == 'holdout') & (frame.cost_bps == 11)]
baseline_name = 'upfront10000_target2_gate3pct' if CONDITIONAL else 'upfront_10000_target8'
baseline = later[later.policy == baseline_name][['ticker', 'entry_date', 'pnl']].rename(columns={'pnl': 'baseline'})
comparisons = []
for policy, group in later.groupby('policy'):
    merged = group.merge(baseline, on=['ticker', 'entry_date'])
    merged['difference'] = merged.pnl - merged.baseline
    comparisons.append({'policy': policy, 'mean_difference_vs_10000_upfront': float(merged.difference.mean()),
                        'ci': clustered_mean_ci(merged, 'difference')})
(ROOT / f'{PREFIX}_comparisons.json').write_text(json.dumps(comparisons, indent=2))
print(pd.DataFrame(summary).query("split == 'holdout' and cost_bps == 11")[[
    'policy', 'n', 'mean', 'win_pct', 'expected_shortfall_5', 'worst', 'loss_over_1500_pct', 'avg_max_capital']].to_string(index=False), flush=True)
print('Coverage:', len(coverage), Counter(e['reason'] for e in excluded), flush=True)
