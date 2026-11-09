"""Capital-matched comparisons, parameter neighborhoods and chronological checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TRIGGERS = [200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200]
CAPS = [500, 1000, 1500, 2000, 3000, 5000, None]
TARGETS = [-500, -250, 0, 250, 500, 1000, 2000, None]


def main():
    descriptions, matrices, upper_matrices, summary_maps = [], [], [], []
    for deadline in ['original', 'close']:
        doc = json.loads((ROOT / f'average_down_grid_{deadline}.json').read_text())
        arrays = np.load(ROOT / f'average_down_grid_{deadline}.npz')
        descriptions.extend([{**p, 'deadline': deadline, 'global_index': len(descriptions) + i} for i, p in enumerate(doc['policies'])])
        matrices.append(arrays['pnl'])
        upper_matrices.append(arrays['upper'])
        summary_maps.append({(int(round(s['cost_bps_each_way'])), s['index']): s for s in doc['summary']})
    pnl = np.concatenate(matrices, axis=1)
    upper = np.concatenate(upper_matrices, axis=1)
    budgets = np.array([p['budget'] for p in descriptions])
    normalized = pnl * 10000 / budgets[None, :, None]
    totals = normalized.sum(axis=2)
    lookup = {(p['deadline'], p['trigger_loss'], tuple(p['adds']), p['target_pnl'], p['loss_cap']): i for i, p in enumerate(descriptions)}
    baseline = lookup[('original', 0, (), None, None)]
    neighborhood = []
    for i, p in enumerate(descriptions):
        keys = set()
        current = (p['deadline'], p['trigger_loss'], tuple(p['adds']), p['target_pnl'], p['loss_cap'])
        keys.add(current)
        for column, levels in [(1, TRIGGERS), (3, TARGETS), (4, CAPS)]:
            if current[column] not in levels:
                continue
            pos = levels.index(current[column])
            for next_pos in [pos-1, pos+1]:
                if 0 <= next_pos < len(levels):
                    candidate = list(current)
                    candidate[column] = levels[next_pos]
                    if tuple(candidate) in lookup:
                        keys.add(tuple(candidate))
        neighborhood.append([lookup[key] for key in keys])

    def smooth(scores):
        return np.array([np.median(scores[neighbors]) for neighbors in neighborhood])

    def detail(i):
        p = descriptions[int(i)]
        source = summary_maps[0 if p['deadline'] == 'original' else 1]
        return {**p, 'cost_scenarios': [source[(bps, p['index'])] for bps in [0,11,26]],
                'normalized_total_by_cost': np.round(totals[:,i],2).tolist(),
                'neighbor_median_net11': round(float(np.median(totals[1,neighborhood[i]])),2),
                'neighbor_min_net11': round(float(totals[1,neighborhood[i]].min()),2),
                'neighbor_max_net11': round(float(totals[1,neighborhood[i]].max()),2),
                'neighbors': len(neighborhood[i])}

    def distinct_top(scores, count=10, predicate=lambda p: True):
        selected, seen = [], set()
        for i in np.argsort(-scores, kind='stable'):
            if not predicate(descriptions[i]):
                continue
            fingerprint = np.round(normalized[1,i], 4).tobytes()
            if fingerprint in seen:
                continue
            selected.append(detail(int(i))); seen.add(fingerprint)
            if len(selected) == count:
                break
        return selected

    winners = {str(bps): detail(int(np.argmax(totals[ci]))) for ci,bps in enumerate([0,11,26])}
    raw_best = int(np.argmax(totals[1]))
    smoothed = smooth(totals[1])
    smooth_best = int(np.argmax(smoothed))
    folds = []
    for start in [12,18,24,30]:
        # Purge the most recent entry date: its next-open exit may coincide
        # with the test period's first opening entry.
        train_end = start - 1
        train_scores = normalized[1,:,:train_end].sum(axis=1)
        raw_choice = int(np.argmax(train_scores))
        stable_choice = int(np.argmax(smooth(train_scores)))
        folds.append({
            'train_dates': [doc['positions'][0]['date'], doc['positions'][train_end-1]['date']],
            'purged_date': doc['positions'][train_end]['date'],
            'test_dates': [doc['positions'][start]['date'], doc['positions'][start+5]['date']],
            'raw_selection': {**descriptions[raw_choice], 'train_pnl_per_10k': round(float(train_scores[raw_choice]),2),
                              'test_pnl_per_10k': round(float(normalized[1,raw_choice,start:start+6].sum()),2)},
            'neighbor_selection': {**descriptions[stable_choice], 'train_pnl_per_10k': round(float(train_scores[stable_choice]),2),
                                   'test_pnl_per_10k': round(float(normalized[1,stable_choice,start:start+6].sum()),2)},
            'baseline_test_pnl_per_10k': round(float(normalized[1,baseline,start:start+6].sum()),2),
        })
    paired = normalized[1,raw_best] - normalized[1,baseline]
    drivers = sorted([{**row, 'improvement_per_10k': round(float(delta),2)} for row,delta in zip(doc['positions'],paired)],
                     key=lambda r: r['improvement_per_10k'],reverse=True)
    grids = []
    for add in [0,2500,5000,10000,20000]:
        for trigger in TRIGGERS:
            for cap in CAPS:
                for target in [-500,-250,0,250,500,1000,2000,None]:
                    i = lookup.get(('original',trigger,(add,),target,cap))
                    if i is not None:
                        grids.append({**descriptions[i], 'gross_per_10k': round(float(totals[0,i]),2),
                                      'net11_per_10k': round(float(totals[1,i]),2),
                                      'net26_per_10k': round(float(totals[2,i]),2),
                                      'delta_vs_baseline_net11': round(float(totals[1,i]-totals[1,baseline]),2)})
    result = {
        'configurations': len(descriptions), 'positions': len(doc['positions']),
        'cost_cases': [0,11,26], 'modeled_price_paths': 2,
        'individual_position_replays': int(np.prod(pnl.shape) * 2),
        'distinct_normalized_net11_return_paths': len(np.unique(np.round(normalized[1],2),axis=0)),
        'baseline': detail(baseline), 'best_by_cost': winners,
        'top_distinct_net11': distinct_top(totals[1], 12),
        'top_distinct_neighborhood_medians': distinct_top(smoothed, 8),
        'best_no_add': distinct_top(totals[1], 3, lambda p: sum(p['adds']) == 0),
        'best_break_even_after_add': distinct_top(totals[1], 3, lambda p: sum(p['adds']) > 0 and p['target_pnl'] == 0),
        'best_accept_small_loss_after_add': distinct_top(totals[1], 3, lambda p: sum(p['adds']) > 0 and p['target_pnl'] is not None and p['target_pnl'] < 0),
        'chronological_folds': folds,
        'chronological_totals_per_10k': {
            'raw_search': round(sum(f['raw_selection']['test_pnl_per_10k'] for f in folds),2),
            'neighbor_search': round(sum(f['neighbor_selection']['test_pnl_per_10k'] for f in folds),2),
            'baseline': round(sum(f['baseline_test_pnl_per_10k'] for f in folds),2)},
        'full_sample_winner_improvement_drivers': drivers,
        'simple_policy_grid': grids,
        'interpretation_limits': [
            'The full sample and many alternatives were already inspected; chronological checks are retrospective diagnostics, not a pristine holdout.',
            'Every strategy is scaled to the same $10K maximum planned capital, including reserved add cash, with loss and target amounts scaled proportionally.',
            'Primary comparison assumes 0.11% cost each way; 0 and 0.26% are sensitivities, not measured execution costs.',
            'The 7,058 configurations are related and often produce identical outcomes; the largest observed score is subject to selection bias.',
            'OHLC fills are idealizations. Second-bar refinements reduce but do not eliminate ordering and liquidity uncertainty.',
            'Daily closed P&L drawdown excludes intraday portfolio marks; per-position worst marks are separately available.',
            'No settings have been changed, no trading orders placed, and no strategy is established as optimal or profitable going forward.',
        ]}
    (ROOT / 'average_down_grid_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    brief=lambda d:{'key':d['key'],'deadline':d['deadline'],'budget':d['budget'],'by_cost_per_10k':d['normalized_total_by_cost'],
                    'neighbor_median':d['neighbor_median_net11'],'neighbor_min':d['neighbor_min_net11']}
    print(json.dumps({
        'configurations':result['configurations'],'unique_paths':result['distinct_normalized_net11_return_paths'],
        'baseline':brief(result['baseline']), 'best':brief(winners['11']),
        'stable_best':brief(detail(smooth_best)),
        'top': [brief(d) for d in result['top_distinct_net11'][:6]],
        'chronological_totals':result['chronological_totals_per_10k'],
        'chronological_choices':[{'test':f['test_dates'],'raw':f['raw_selection']['key'],'stable':f['neighbor_selection']['key']} for f in folds],
        'top_improvement_drivers':drivers[:5],
    },indent=2))


if __name__ == '__main__':
    main()
