"""Zero-cost score/outcome audit using recorded, contemporaneous Rank scores.

No retraining, hindsight rescoring, broker mutations or production edits.
The documented 2026-09-26 objective change defines the principal era check.
Saved top-20 rows are complete, permitting balanced day/rank demeaning.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median, pstdev

import numpy as np

from average_down_search_engine import SearchPolicy, replay
from search_average_down import load_paths

ROOT = Path(__file__).resolve().parent
NEW_OBJECTIVE_FIRST_SESSION = '2026-09-28'
ASSUMED_COST = 0.0
BOOTSTRAPS = 3000
FEATURES = ['score', 'gap_to_rank2', 'top20_z', 'gap_to_rank2_z']


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(x*x) * np.sum(y*y)))
    return float(np.sum(x*y) / denominator) if denominator > 1e-14 else None


def ranks(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = np.empty(len(values))
    lo = 0
    while lo < len(order):
        hi = lo + 1
        while hi < len(order) and values[order[hi]] == values[order[lo]]:
            hi += 1
        result[order[lo:hi]] = (lo + 1 + hi) / 2
        lo = hi
    return result


def percentile(values, q):
    ordered = sorted(v for v in values if v is not None)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi-position) + ordered[hi] * (position-lo) if hi != lo else ordered[lo]


def week_resamples(dates, seed=20261107):
    weeks = defaultdict(list)
    for i, day in enumerate(dates):
        weeks[date.fromisoformat(day).isocalendar()[:2]].append(i)
    groups = list(weeks.values())
    rng = random.Random(seed)
    samples = [[i for _ in groups for i in rng.choice(groups)] for _ in range(BOOTSTRAPS)]
    return len(groups), samples


def correlated(rows, feature, outcome):
    x = np.array([r[feature] for r in rows]); y = np.array([r[outcome] for r in rows])
    blocks, samples = week_resamples([r['date'] for r in rows])
    pearson, spearman = [], []
    for sample in samples:
        pearson.append(correlation(x[sample], y[sample]))
        spearman.append(correlation(ranks(x[sample]), ranks(y[sample])))
    return {'n': len(rows), 'week_blocks': blocks, 'feature': feature, 'outcome': outcome,
            'pearson': correlation(x, y), 'spearman': correlation(ranks(x), ranks(y)),
            'pearson_week_bootstrap_95': [percentile(pearson, .025), percentile(pearson, .975)],
            'spearman_week_bootstrap_95': [percentile(spearman, .025), percentile(spearman, .975)]}


def groups(rows):
    result = []
    for label, low, high in [('below_0.4', float('-inf'), .4), ('0.4_to_below_0.6', .4, .6), ('0.6_and_above', .6, float('inf'))]:
        selected = [r for r in rows if low <= r['score'] < high]
        result.append({'score_group': label, 'n': len(selected),
                       'mean_strategy_return_pct': 100 * mean(r['strategy_return'] for r in selected) if selected else None,
                       'mean_strategy_pnl_10k': round(mean(r['strategy_pnl_10k'] for r in selected), 2) if selected else None,
                       'total_strategy_pnl_10k': round(sum(r['strategy_pnl_10k'] for r in selected), 2),
                       'median_strategy_pnl_10k': round(median(r['strategy_pnl_10k'] for r in selected), 2) if selected else None,
                       'wins': sum(r['strategy_return'] > 0 for r in selected),
                       'mean_close_return_pct': 100 * mean(r['close_return'] for r in selected) if selected else None})
    return result


def two_way_residuals(matrix):
    """Remove date and rank fixed effects in a complete, balanced panel."""
    return matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + matrix.mean()


def panel_stats(days):
    scores = np.array([[r['score'] for r in day] for day in days])
    returns = np.array([[r['close_return'] for r in day] for day in days])
    ic = [correlation(ranks(x), ranks(y)) for x, y in zip(scores, returns)]
    blocks, samples = week_resamples([day[0]['date'] for day in days])
    means, extras = [], []
    for indexes in samples:
        means.append(mean(ic[i] for i in indexes))
        extras.append(correlation(two_way_residuals(scores[indexes]).ravel(), two_way_residuals(returns[indexes]).ravel()))
    return {'days': len(days), 'observations': int(scores.size), 'week_blocks': blocks,
            'mean_within_day_spearman': mean(ic),
            'mean_within_day_spearman_week_bootstrap_95': [percentile(means, .025), percentile(means, .975)],
            'raw_score_close_pearson_pooled': correlation(scores.ravel(), returns.ravel()),
            'score_close_partial_pearson_after_date_and_rank': correlation(two_way_residuals(scores).ravel(), two_way_residuals(returns).ravel()),
            'partial_pearson_week_bootstrap_95': [percentile(extras, .025), percentile(extras, .975)],
            'daily_ic': [{'date': day[0]['date'], 'value': value} for day, value in zip(days, ic)]}


def main():
    scans = json.loads((ROOT / 'rank_scan_snapshot.json').read_text())
    paper = json.loads((ROOT / 'live_repaired_api_snapshot.json').read_text())
    frozen = json.loads((ROOT / 'rank1_significance_rows.json').read_text())
    overnight = json.loads((ROOT / 'overnight_actuals_snapshot.json').read_text())
    source1 = {r['date']: r for r in frozen}
    records, panels, top2, missing = [], [], [], []
    for day in sorted(paper['days'], key=lambda d: d['as_of']):
        if day.get('scan_id') or not day.get('rank'):
            continue
        as_of = day['as_of']; signals = scans[as_of]['signals']; positions = day['rank']
        assert len(signals) == len(positions) == 20
        scores = [s['predicted_return'] for s in signals]
        spread = pstdev(scores)
        next_prices = {r['ticker']: r for r in overnight[as_of]['rows']}
        panel = []
        for position, signal in zip(positions, signals):
            assert position['ticker'] == signal['ticker']
            assert abs(position['predicted'] - signal['predicted_return']) < 1e-10
            close_return = position['close_price'] / position['open_price'] - 1
            assert abs(close_return - position['session_return']) < 1e-8
            row = {'date': as_of, 'ticker': position['ticker'], 'rank': position['rank'],
                   'score': position['predicted'], 'entry': position['open_price'], 'close': position['close_price'],
                   'close_return': close_return, 'top20_z': (position['predicted']-mean(scores))/spread,
                   'reported_model_run_id': scans[as_of].get('model_run_id'),
                   'reported_trained_through': scans[as_of].get('freshness', {}).get('model_trained_through')}
            panel.append(dict(row))
            if position['rank'] <= 2:
                following = next_prices[row['ticker']].get('next_open')
                if close_return < 0 and not following:
                    missing.append({'date': as_of, 'ticker': row['ticker'], 'rank': row['rank'], 'reason': next_prices[row['ticker']]['status']})
                    continue
                exit_price = following if close_return < 0 else row['close']
                row['exit'] = exit_price
                row['strategy_pnl_10k'] = round(10000 * (exit_price / row['entry'] - 1), 2)
                row['price_recomputed_pnl_10k'] = row['strategy_pnl_10k']
                if position['rank'] == 1:
                    # Keep the frozen dashboard-audit cent rounding after
                    # independently checking price-based cashflow. Three
                    # records differ by one cent from recomputation.
                    assert abs(row['strategy_pnl_10k'] - source1[as_of]['pnl']) < .011
                    row['strategy_pnl_10k'] = source1[as_of]['pnl']
                row['strategy_return'] = row['strategy_pnl_10k'] / 10000
                top2.append(dict(row))
                if position['rank'] == 1:
                    assert row['ticker'] == source1[as_of]['ticker']
                    assert abs(row['strategy_pnl_10k'] - source1[as_of]['pnl']) < .011
                    row['gap_to_rank2'] = scores[0] - scores[1]
                    row['gap_to_rank2_z'] = row['gap_to_rank2'] / spread
                    records.append(row)
        panels.append(panel)
    assert len(records) == 36 and round(sum(r['strategy_pnl_10k'] for r in records), 2) == 2708.87
    assert sum(map(len, panels)) == 720
    _, paths = load_paths('original')
    add_ledger, gate_results = [], defaultdict(list)
    for row, alternatives in zip(records, paths):
        assert alternatives[0].original['date'] == row['date']
        row['intraday_worst_return'] = float(alternatives[0].prices[:alternatives[0].add_until].min() / row['entry'] - 1)
        add_only = [replay(path, SearchPolicy(500, (5000,), None, None), ASSUMED_COST) for path in alternatives]
        if add_only[0]['entries']:
            first = add_only[0]['entries'][0]
            recovery = sorted([replay(path, SearchPolicy(500, (5000,), 0, None), ASSUMED_COST) for path in alternatives], key=lambda r: r['pnl'])
            add_ledger.append({**row, 'add_price': first['price'], 'add_at': first['at'],
                               'post_drop_return_to_original_exit': row['exit'] / first['price'] - 1,
                               'reached_partial_add_break_even': recovery[0]['reason'].startswith('target'),
                               'recovery_path_ambiguous': recovery[0]['pnl'] != recovery[1]['pnl']})
        for label, gate in [('all_scores', None), ('score_at_least_0.4', .4), ('score_at_least_0.6', .6)]:
            allow_add = gate is None or row['score'] >= gate
            policy = SearchPolicy(500, (5000,) if allow_add else (), None, 2000)
            outcome = min((replay(path, policy, ASSUMED_COST) for path in alternatives), key=lambda r: r['pnl'])
            gate_results[label].append({'date': row['date'], 'pnl': outcome['pnl'], 'adds': outcome['adds'], 'score': row['score']})
    eras = {}
    for label, boundary in [('all_dates', '0000'), ('after_documented_objective_change', NEW_OBJECTIVE_FIRST_SESSION)]:
        selected = [r for r in records if r['date'] >= boundary]
        selected_days = [day for day in panels if day[0]['date'] >= boundary]
        dropped = [r for r in add_ledger if r['date'] >= boundary]
        dropped_bins = []
        for low, high in [(float('-inf'), .4), (.4, .6), (.6, float('inf'))]:
            subset = [r for r in dropped if low <= r['score'] < high]
            dropped_bins.append({'score_low': low if low != float('-inf') else None, 'score_high': high if high != float('inf') else None,
                                 'n': len(subset), 'recovered_partial_add_be': sum(r['reached_partial_add_break_even'] for r in subset),
                                 'mean_post_drop_return_pct': 100 * mean(r['post_drop_return_to_original_exit'] for r in subset) if subset else None})
        correlations = [correlated(selected, feature, outcome) for feature in FEATURES for outcome in ['strategy_return', 'close_return']]
        correlations.append(correlated(selected, 'score', 'intraday_worst_return'))
        correlations.append(correlated(dropped, 'score', 'post_drop_return_to_original_exit'))
        rank2 = [r for r in top2 if r['rank'] == 2 and r['date'] >= boundary]
        eras[label] = {'n': len(selected), 'dates': [selected[0]['date'], selected[-1]['date']],
                       'rank1_score_groups': groups(selected), 'rank2_score_groups': groups(rank2),
                       'rank1_correlations': correlations,
                       'rank2_raw_score_strategy': correlated(rank2, 'score', 'strategy_return'),
                       'top20_panel': panel_stats(selected_days),
                       'drawdown_n': len(dropped), 'drawdown_score_groups': dropped_bins,
                       'entry_gate_results': [{'threshold': threshold,
                                               'trades': sum(r['score'] >= threshold for r in selected),
                                               'pnl': round(sum(r['strategy_pnl_10k'] for r in selected if r['score'] >= threshold), 2)} for threshold in [.4, .6]],
                       'add_gate_results': [{'gate': key, 'trades': len(selected),
                                             'added_positions': sum(r['adds'] > 0 for r in rows if r['date'] >= boundary),
                                             'pnl_10k_initial_5k_reserve': round(sum(r['pnl'] for r in rows if r['date'] >= boundary), 2),
                                             'pnl_per_10k_maximum': round(sum(r['pnl'] for r in rows if r['date'] >= boundary) * 2/3, 2)} for key, rows in gate_results.items()]}
    zero_grid = []
    for deadline in ['original', 'close']:
        doc = json.loads((ROOT / f'average_down_grid_{deadline}.json').read_text())
        zero_grid.extend([{**r, 'deadline': deadline} for r in doc['summary'] if r['cost_bps_each_way'] == 0])
    best = max(zero_grid, key=lambda r: r['pnl_on_10k_reserved_budget'])
    best_no_add = max((r for r in zero_grid if not sum(r['adds'])), key=lambda r: r['pnl_on_10k_reserved_budget'])
    candidate = next(r for r in zero_grid if r['deadline'] == 'original' and r['key'] == 'at500_adds5000_targetNone_stop2000')
    assert abs(eras['all_dates']['add_gate_results'][0]['pnl_10k_initial_5k_reserve'] - candidate['pnl']) < .011
    # Confirm score normalization removes an arbitrary shift of raw scores.
    raw = np.array([1.0, .6, .4, .1]); shifted = raw + .27
    assert abs(correlation(raw, shifted) - 1) < 1e-12
    assert np.allclose((raw-raw.mean()) / raw.std(), (shifted-shifted.mean()) / shifted.std())
    for x, expected in [([4, 2, 2, 1], [4, 2.5, 2.5, 1]), ([1, 2, 3], [1, 2, 3])]:
        assert list(ranks(x)) == expected
    result = {'assumed_cost_each_way': ASSUMED_COST, 'original_pnl_10k': 2708.87,
              'recorded_scores_reconciled': 720, 'rank1_rows': records, 'top2_closed_rows': top2,
              'top2_missing_exit': missing, 'top20_rows': [r for day in panels for r in day],
              'eras': eras, 'drawdown_ledger': add_ledger,
              'zero_cost_existing_grid': {'original_pnl_per_10k_maximum': 2708.87, 'preselected_partial_add': candidate,
                                          'best_after_inspecting_full_sample': best, 'best_no_add_after_inspecting_full_sample': best_no_add},
              'method': [
                  'Use recorded morning scores, not a current model rescored on historical dates. Exclude both October 28 replays.',
                  'Rank1 strategy returns reconcile to the 36-position corrected dashboard study. Top20 close returns reconcile to all 720 repaired live paper rows.',
                  'Three Rank1 records differ by one cent from direct price recomputation; the prior frozen dashboard-audit rounding is preserved, and recomputed amounts are retained alongside it.',
                  'Top2 next-open exits use verified snapshots; missing required exits remain excluded and listed, never invented.',
                  'The primary era boundary follows the documented September 26 objective change; it is not exact model-version stratification. Model retraining and feature changes continue within the later period.',
                  'model_run_id in morning payloads is not a reliable Rank-artifact identifier: the morning path independently loads RANK_MODEL_NAME.',
                  'Raw scores are relative listwise outputs. The loss is invariant to a common shift within a list; the scores are not calibrated probabilities or return percentages.',
                  'Standardization uses only the recorded top20 list, not the full scored universe, which is unavailable in these snapshots.',
                  'Pearson measures a linear association; Spearman measures a monotone association. Neither establishes a trading edge or causation.',
                  '95% sensitivity intervals resample whole trading weeks, 3000 times. Only 8 total weeks and 6 later weeks exist; inference is fragile.',
                  'Top20 partial correlations remove date and ordinal-rank fixed effects by two-way demeaning of the balanced panel. This is an exploratory residual association, not held-out prediction.',
                  'Score bins 0.4 and 0.6 reflect the user question. Other diagnostics and gate outcomes are exploratory and unadjusted for multiple comparisons.',
                  'After-drop outcomes start at the first modeled 5% add fill, use the existing close/next-open deadline, and evaluate the original morning score. The model was not rescored at the drop.',
                  'Cost assumption is exactly zero. Prices remain aggregate-based hypothetical fills, not a promise of executable prices; opening entries can precede published picks.',
                  'No model, dashboard, live order or production setting is changed by this study.',
              ],
              'checks': {'rank1_baseline_reconciled': True, 'scores_and_close_returns_reconciled': 720,
                         'preselected_add_matches_zero_cost_grid': True, 'score_shift_and_tie_ranks': True},
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                ['rank_scan_snapshot.json', 'live_repaired_api_snapshot.json', 'rank1_significance_rows.json', 'overnight_actuals_snapshot.json']}}
    (ROOT / 'rank_score_study.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    brief = {}
    for key, value in eras.items():
        brief[key] = {k:value[k] for k in ['n', 'rank1_score_groups', 'drawdown_n', 'drawdown_score_groups', 'add_gate_results']}
        brief[key]['score_correlations'] = [r for r in value['rank1_correlations'] if r['feature'] == 'score']
        brief[key]['top20'] = {k:v for k,v in value['top20_panel'].items() if k != 'daily_ic'}
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
