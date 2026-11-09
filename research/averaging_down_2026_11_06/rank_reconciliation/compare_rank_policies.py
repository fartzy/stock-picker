"""Matched exploratory comparison; original saved picks, no production mutations.

The primary policy comparison uses identical entries with ten completed sessions,
adequate minute coverage, and compatible price bases. Newer entries are excluded
by date before examining policy outcomes, rather than force-liquidated early.
Publication and clock-entry results are sensitivity checks on the final saved
picks, not a reconstruction of which picks were visible at every timestamp.
"""
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import gzip
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from minute_data import regular_frame
from simulation import Policy, simulate, clustered_mean_ci

rows = json.loads((ROOT / 'dashboard_rows.json').read_text())
VERIFIED = '--verified-prices' in sys.argv
PREFIX = 'verified_' if VERIFIED else ''
if VERIFIED:
    verified_rows = {(r['date'], r['ticker']): r for r in json.loads((ROOT / 'verified_price_rows.json').read_text())}
    rows = [{**r, 'entry_price': verified_rows[(r['date'], r['ticker'])]['verified_open'],
             'close_price': verified_rows[(r['date'], r['ticker'])]['verified_close'],
             'held_overnight': verified_rows[(r['date'], r['ticker'])]['held_overnight_after_verification'],
             'display_total_pnl': verified_rows[(r['date'], r['ticker'])]['pnl']}
            if (r['date'], r['ticker']) in verified_rows else r for r in rows]
publications = {r['date']: r for r in json.loads((ROOT / 'publication_times.json').read_text())}
daily = pd.read_pickle(ROOT.parent / 'five_year_arrays.pkl')
calendar = daily['SPY']['dates']
POLICIES = [
    (Policy('single_2500_target2_gate3pct', loss_limit=1200, target=.02,
            overnight_loss_limit=600, overnight_loss_fraction=.03), 2500),
    (Policy('ladder_2500_target2_gate3pct', max_tranches=4, loss_limit=1200, target=.02,
            overnight_loss_limit=600, overnight_loss_fraction=.03), 2500),
    (Policy('upfront_10000_target2_gate3pct', initial_tranches=4, max_tranches=4,
            loss_limit=1200, target=.02, overnight_loss_limit=600, overnight_loss_fraction=.03), 2500),
]
coverage, excluded, results, timing, basis = [], [], [], [], []


def fixed_result(entry, exit_price, capital, cost):
    return capital * (exit_price / entry * (1 - cost) / (1 + cost) - 1)


def gate_slots(frame):
    flags = np.zeros(len(frame), dtype=bool)
    dates = frame.index.date
    for day in sorted(set(dates)):
        slots = []
        for i in np.flatnonzero(dates == day):
            minute = frame.index[i].hour * 60 + frame.index[i].minute
            if (950 <= minute <= 958 and i + 1 < len(frame) and dates[i + 1] == day
                    and (frame.index[i + 1] - frame.index[i]).total_seconds() <= 120):
                slots.append((abs(minute - 955), i))
        if not slots:
            return None
        first = min(slots)[1]
        for _, i in slots:
            if i >= first:
                flags[i] = True
    return flags


for row in rows:
    key = {k: row[k] for k in ['date', 'ticker', 'rank']}
    path = next((ROOT / 'minute_bars').glob(f"{row['ticker']}_{row['date']}_*.json.gz"))
    with gzip.open(path, 'rt') as handle:
        frame = regular_frame(json.load(handle))
    if frame.empty or frame.index.has_duplicates:
        excluded.append({**key, 'reason': 'empty_or_duplicate_minutes'})
        continue
    first_day = frame.loc[row['date']]
    basis.append({**key, 'book_open': row['entry_price'], 'minute_open': float(first_day.iloc[0].o),
        'first_bar': first_day.index[0].isoformat(), 'book_close': row['close_price'],
        'minute_close': float(first_day.iloc[-1].c), 'bars_first_day': len(first_day),
        'open_difference_pct': 100 * (first_day.iloc[0].o / row['entry_price'] - 1)})
    if not row['included_in_headline']:
        excluded.append({**key, 'reason': 'dashboard_day_pending'})
        continue
    # Keep close and next-open prices unchanged; vary only entry time and
    # re-evaluate whether the position is losing at that day's close.
    for label, threshold in [('09:35', f"{row['date']}T09:35:00"),
                             ('09:40', f"{row['date']}T09:40:00"),
                             ('publication', publications.get(row['date'], {}).get('published_at_et'))]:
        if threshold is None:
            continue
        stamp = pd.Timestamp(threshold)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize('America/New_York')
        selected = first_day[first_day.index >= stamp]
        if selected.empty or (selected.index[0] - stamp).total_seconds() > 120:
            continue
        entry = float(selected.iloc[0].o)
        held = row['close_price'] < entry
        exit_price = row['next_open'] if held else row['close_price']
        if exit_price is None:
            continue
        pnl = fixed_result(entry, exit_price, 10000, 0)
        timing.append({**key, 'entry_mode': label, 'entry_at': selected.index[0].isoformat(),
                       'entry_price': entry, 'held_overnight': held, 'pnl': pnl,
                       'saved_pnl': row['display_total_pnl'],
                       'net11': fixed_result(entry, exit_price, 10000, .0011)})
    deadline = date.fromisoformat(row['date']) + timedelta(days=14)
    expected = [d for d in calendar if row['date'] <= d < deadline.isoformat()][:10]
    if len(expected) < 10:
        excluded.append({**key, 'reason': 'ten_sessions_not_yet_observed'})
        continue
    groups = {str(day): group for day, group in frame.groupby(frame.index.date)}
    if sorted(groups) != expected:
        excluded.append({**key, 'reason': 'missing_session'})
        continue
    valid_coverage = all(len(g) >= 180 and (g.index[0].hour * 60 + g.index[0].minute) <= 575
                         and (g.index[-1].hour * 60 + g.index[-1].minute) >= 955 for g in groups.values())
    if not valid_coverage:
        excluded.append({**key, 'reason': 'sparse_minutes'})
        continue
    if abs(first_day.iloc[0].o / row['entry_price'] - 1) > .02:
        excluded.append({**key, 'reason': 'saved_open_differs_more_than_2pct'})
        continue
    if abs(first_day.iloc[-1].c / row['close_price'] - 1) > .02:
        excluded.append({**key, 'reason': 'saved_close_differs_more_than_2pct'})
        continue
    flags = gate_slots(frame)
    if flags is None:
        excluded.append({**key, 'reason': 'no_same_session_overnight_review_fill'})
        continue
    bars = frame[['o', 'h', 'l', 'c', 'v']].to_numpy(dtype=float)
    times = frame.index.as_unit('s').asi8.astype(float)
    assert np.isfinite(bars).all() and (bars[:, :4] > 0).all()
    assert (bars[:, 1] + 1e-5 >= np.maximum(bars[:, 0], bars[:, 3])).all()
    assert (bars[:, 2] - 1e-5 <= np.minimum(bars[:, 0], bars[:, 3])).all()
    exit_price = row['next_open'] if row['held_overnight'] else row['close_price']
    if exit_price is None:
        excluded.append({**key, 'reason': 'missing_baseline_exit'})
        continue
    coverage.append({**key, 'bars': len(frame), 'sessions': len(groups), 'end_date': expected[-1]})
    for cost_bps in [0, 11, 26]:
        for capital in [2500, 10000]:
            results.append({**key, 'cost_bps': cost_bps, 'policy': f'dashboard_{capital}',
                'pnl': fixed_result(row['entry_price'], exit_price, capital, cost_bps / 10000),
                'max_capital': capital, 'adds': 0, 'exit_reason': 'next_open' if row['held_overnight'] else 'close'})
        for policy, tranche in POLICIES:
            result = simulate(bars, times, np.zeros(len(frame), dtype=bool), policy,
                cost=cost_bps / 10000, tranche=tranche, entry_price=row['entry_price'], overnight_check=flags)
            results.append({**key, 'cost_bps': cost_bps, 'policy': policy.name, **result})
    if len(coverage) % 10 == 0:
        print('Matched', len(coverage), 'positions', flush=True)

f = pd.DataFrame(results)
f.to_parquet(ROOT / f'{PREFIX}matched_policy_results.parquet', index=False)
summary = []
for (cost, policy), group in f.groupby(['cost_bps', 'policy']):
    x = group.pnl.to_numpy()
    summary.append({'cost_bps': int(cost), 'policy': policy, 'n': len(x),
        'total_pnl': float(x.sum()), 'mean_per_position': float(x.mean()),
        'median': float(np.median(x)), 'win_pct': float((x > 0).mean() * 100),
        'worst': float(x.min()), 'best': float(x.max()),
        'average_peak_capital_per_position': float(group.max_capital.mean()),
        'average_adds': float(group['adds'].mean()),
        'average_duration_days': float(group.duration_days.mean()) if group.duration_days.notna().any() else None,
        'exit_counts': dict(Counter(group.exit_reason))})
t = pd.DataFrame(timing)
timing_summary = []
for label, group in t.groupby('entry_mode'):
    timing_summary.append({'entry_mode': label, 'n': len(group),
        'first_date': group.date.min(), 'last_date': group.date.max(),
        'saved_total_pnl_same_positions': float(group.saved_pnl.sum()),
        'delayed_total_pnl': float(group.pnl.sum()), 'delayed_mean': float(group.pnl.mean()),
        'delayed_net11_total': float(group.net11.sum()), 'delayed_net11_mean': float(group.net11.mean()),
        'win_pct': float((group.pnl > 0).mean() * 100)})
paired = []
base = f[(f.cost_bps == 11) & f.policy.eq('dashboard_2500')][['date', 'ticker', 'pnl']].rename(columns={'pnl': 'baseline'})
for policy in ['single_2500_target2_gate3pct', 'ladder_2500_target2_gate3pct']:
    group = f[(f.cost_bps == 11) & f.policy.eq(policy)].merge(base, on=['date', 'ticker'])
    group['difference'] = group.pnl - group.baseline
    group['entry_date'] = group['date']
    paired.append({'policy': policy, 'n': len(group), 'mean_difference_vs_dashboard2500': float(group.difference.mean()),
                   'cluster_bootstrap_95pct': clustered_mean_ci(group, 'difference')})
out = {'entry_basis': 'verified daily open' if VERIFIED else 'original saved open',
       'coverage': coverage, 'excluded': excluded, 'exclusion_counts': dict(Counter(r['reason'] for r in excluded)),
       'policies': summary, 'timing_sensitivity': timing_summary, 'paired': paired}
(ROOT / f'{PREFIX}matched_summary.json').write_text(json.dumps(out, indent=2))
(ROOT / f'{PREFIX}minute_basis_check.json').write_text(json.dumps(basis, indent=2))
(ROOT / f'{PREFIX}timing_sensitivity_rows.json').write_text(json.dumps(timing, indent=2))
print(json.dumps({'n_matched': len(coverage), 'excluded': out['exclusion_counts'],
                  'policies_11bps': [r for r in summary if r['cost_bps'] == 11],
                  'timing': timing_summary, 'paired': paired}, indent=2))
