"""Run a finite theoretical search; save every result, not just the winner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from average_down_search_engine import policy_grid, price_path, replay
from rank1_average_down_study import refine_bars
from rank1_stop_loss_study import regular_bars

ROOT = Path(__file__).resolve().parent
COSTS = [0.0, .0011, .0026]


def load_paths(deadline):
    rows = json.loads((ROOT / 'rank1_significance_rows.json').read_text())
    actuals = json.loads((ROOT / 'overnight_actuals_snapshot.json').read_text())
    paths = []
    for row in rows:
        row['next_session'] = next(x['next_session'] for x in actuals[row['date']]['rows'] if x['ticker'] == row['ticker'])
        with gzip.open(next((ROOT / 'minute_bars').glob(f"{row['ticker']}_{row['date']}_*")), 'rt') as handle:
            doc = json.load(handle)
        bars, _ = refine_bars(row['ticker'], regular_bars(doc['bars'], row['date']))
        paths.append([price_path(row, bars, order, deadline) for order in ['OHLC', 'OLHC']])
    return rows, paths


def stats(values):
    curve = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0, curve])[1:]
    return {'pnl': round(float(values.sum()), 2), 'worst_trade': float(values.min()),
            'max_closed_drawdown': round(float((peak - curve).max()), 2),
            'wins': int((values > 0).sum()), 'losses': int((values < 0).sum()), 'flats': int((values == 0).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', choices=['original', 'close'], required=True)
    args = parser.parse_args()
    rows, paths = load_paths(args.deadline)
    policies = policy_grid()
    shape = (len(COSTS), len(policies), len(rows))
    pnl, upper, worst, capital = [np.zeros(shape) for _ in range(4)]
    adds = np.zeros(shape, dtype=np.int8)
    metadata = [{'index': i, 'key': policy.key, 'budget': policy.budget, **asdict(policy)} for i, policy in enumerate(policies)]
    summaries = []
    start = time.monotonic()
    for ci, cost in enumerate(COSTS):
        for pi, policy in enumerate(policies):
            for di, alternatives in enumerate(paths):
                outcomes = sorted([replay(path, policy, cost) for path in alternatives], key=lambda o: (o['pnl'], o['worst_mark']))
                lo, hi = outcomes
                pnl[ci, pi, di], upper[ci, pi, di] = lo['pnl'], hi['pnl']
                worst[ci, pi, di], capital[ci, pi, di], adds[ci, pi, di] = lo['worst_mark'], lo['capital'], lo['adds']
            summary = {**metadata[pi], 'cost_bps_each_way': cost * 10000, **stats(pnl[ci, pi]),
                       'pnl_on_10k_reserved_budget': round(float(pnl[ci, pi].sum()) * 10000 / policy.budget, 2),
                       'pnl_upper_path': round(float(upper[ci, pi].sum()), 2),
                       'ambiguous_positions': int((upper[ci, pi] != pnl[ci, pi]).sum()),
                       'worst_mark': float(worst[ci, pi].min()), 'avg_capital': round(float(capital[ci, pi].mean()), 2),
                       'positions_added': int((adds[ci, pi] > 0).sum()), 'total_adds': int(adds[ci, pi].sum()),
                       'thirds_pnl': [round(float(pnl[ci, pi, a:b].sum()), 2) for a, b in [(0,12),(12,24),(24,36)]]}
            summaries.append(summary)
            if (pi + 1) % 1000 == 0:
                print(args.deadline, 'cost', cost, 'policies', pi + 1, '/', len(policies), 'seconds', round(time.monotonic()-start,1), flush=True)
    np.savez_compressed(ROOT / f'average_down_grid_{args.deadline}.npz', pnl=pnl, upper=upper, worst=worst, capital=capital, adds=adds)
    output = {'deadline': args.deadline, 'positions': [{'date': row['date'], 'ticker': row['ticker']} for row in rows],
              'costs': COSTS, 'policies': metadata, 'summary': summaries,
              'policy_count': len(policies), 'price_orderings': ['OHLC', 'OLHC'],
              'elapsed_seconds': round(time.monotonic()-start, 2),
              'meaning': 'Retrospective theoretical search, not a held-out proof of an optimal policy. Prices and execution assumptions match the documented one-add study.'}
    (ROOT / f'average_down_grid_{args.deadline}.json').write_text(json.dumps(output, indent=2) + '\n')
    primary = sorted([s for s in summaries if s['cost_bps_each_way'] == 11], key=lambda s: s['pnl_on_10k_reserved_budget'], reverse=True)
    print('DONE',args.deadline,'policies',len(policies),'seconds',output['elapsed_seconds'],flush=True)
    print(json.dumps(primary[:8],indent=2),flush=True)


if __name__ == '__main__':
    main()
