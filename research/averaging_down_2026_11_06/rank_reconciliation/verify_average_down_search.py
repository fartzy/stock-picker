"""Independent one-add parity, synthetic gaps, and saved-grid invariants."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from average_down_search_engine import PricePath, SearchPolicy, price_path, replay
from rank1_average_down_study import Policy, refine_bars, replay as reference_replay
from rank1_stop_loss_study import regular_bars

ROOT = Path(__file__).resolve().parent


def synthetic(prices, gaps, *, entry=100):
    row = {'entry': entry, 'pnl': round(10000 * (prices[-1] / entry - 1), 2)}
    return PricePath(np.array(prices, dtype=float), np.array(gaps),
                     [str(i) for i in range(len(prices))], len(prices) - 1,
                     row, 'original', 'synthetic')


def main():
    path = synthetic([100, 95, 95], [True, False, True])
    outcome = replay(path, SearchPolicy(500, (10000,), None, None))
    assert outcome['pnl'] == -500 and outcome['capital'] == 20000
    gap = synthetic([100, 80, 92], [True, True, True])
    outcome = replay(gap, SearchPolicy(500, (5000, 5000), None, None))
    assert outcome['pnl'] == 700 and [e['price'] for e in outcome['entries']] == [80, 80]
    stopped = replay(gap, SearchPolicy(500, (5000, 5000), None, 1500))
    assert stopped['pnl'] == -2000 and stopped['adds'] == 0
    # After adding at 95, the order of a later high and low determines which
    # resting exit executes first; neither path is silently preferred.
    target_first = synthetic([100, 95, 94, 100, 85, 95], [True, False, True, False, False, True])
    stop_first = synthetic([100, 95, 94, 85, 100, 95], [True, False, True, False, False, True])
    policy = SearchPolicy(500, (10000,), 0, 1500)
    assert replay(target_first, policy)['pnl'] == 0
    assert replay(stop_first, policy)['pnl'] == -1500
    net = replay(target_first, SearchPolicy(500, (10000,), 0, None), .0011)
    qty = 10000 / (100 * 1.0011) + 10000 / (95 * 1.0011)
    assert abs(qty * net['exit_price'] * .9989 - 20000) < 1e-8
    assert net['exit_price'] > 20000 / (100 + 10000 / 95)

    rows = json.loads((ROOT / 'rank1_significance_rows.json').read_text())
    actuals = json.loads((ROOT / 'overnight_actuals_snapshot.json').read_text())
    cases = [(500, 10000, None, True), (500, 5000, 2000, False),
             (400, 2500, 1500, True), (1000, 0, 1500, False),
             (600, 10000, 1500, True), (700, 10000, 1000, True)]
    checked = 0
    for row in rows:
        row['next_session'] = next(r['next_session'] for r in actuals[row['date']]['rows'] if r['ticker'] == row['ticker'])
        with gzip.open(next((ROOT / 'minute_bars').glob(f"{row['ticker']}_{row['date']}_*")), 'rt') as handle:
            document = json.load(handle)
        bars, _ = refine_bars(row['ticker'], regular_bars(document['bars'], row['date']))
        for order in ['OHLC', 'OLHC']:
            path = price_path(row, bars, order)
            for cost in [0, .0011, .0026]:
                for trigger, add, cap, break_even in cases:
                    new = replay(path, SearchPolicy(trigger, (add,), 0 if break_even else None, cap), cost)
                    old = reference_replay(row, bars, Policy(trigger, add, cap, break_even), cost, order)
                    assert abs(new['pnl'] - old['pnl']) < .021, (row['ticker'], order, cost, new, old)
                    qty = 10000 / (row['entry'] * (1 + cost))
                    qty += sum(e['cash'] / (e['price'] * (1 + cost)) for e in new['entries'])
                    expected = qty * new['exit_price'] * (1 - cost) - new['capital']
                    assert abs(expected - new['pnl']) < .021
                    checked += 1
    array_checks = 0
    for deadline in ['original', 'close']:
        doc = json.loads((ROOT / f'average_down_grid_{deadline}.json').read_text())
        matrix = np.load(ROOT / f'average_down_grid_{deadline}.npz')
        assert matrix['pnl'].shape == (3, 3529, 36)
        assert np.isfinite(matrix['pnl']).all() and np.isfinite(matrix['worst']).all()
        assert (matrix['pnl'] <= matrix['upper'] + .001).all()
        assert (matrix['worst'] <= matrix['pnl'] + .021).all()
        budgets = np.array([p['budget'] for p in doc['policies']])[None, :, None]
        assert (matrix['capital'] <= budgets).all()
        for summary in doc['summary']:
            ci = [0, 11, 26].index(round(summary['cost_bps_each_way']))
            values = matrix['pnl'][ci, summary['index']]
            assert abs(values.sum() - summary['pnl']) < .011
            assert abs(values.min() - summary['worst_trade']) < .001
            array_checks += 1
    filenames = ['average_down_search_engine.py', 'search_average_down.py',
                 'rank1_significance_rows.json', 'average_down_grid_original.npz',
                 'average_down_grid_close.npz', 'average_down_grid_analysis.json',
                 'average_down_historical_stress.json']
    output = {'synthetic_cases': 6, 'independent_reference_and_cashflow_checks': checked,
              'saved_summary_reconciliations': array_checks,
              'array_shape_each_deadline': [3, 3529, 36], 'status': 'passed',
              'sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in filenames}}
    (ROOT / 'average_down_grid_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
