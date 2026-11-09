"""Stress fixed Rank-search candidates on the existing broad historical sample.

This is not another parameter search or a validation of the Rank model. The
sample contains liquid stocks following a prior-day decline of at least 5%,
using the 433 episodes that passed the earlier, unchanged coverage checks.
All policies use the same minute-based opening/closing prices and reserve-
capital normalization. Overlapping episodes are not an account equity curve.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

from average_down_search_engine import SearchPolicy, price_path, replay
from rank1_stop_loss_study import local_time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
# Fixed before examining this follow-up sample. No winners chosen from it.
POLICIES = {
    'original': SearchPolicy(0, (), None, None),
    'no_add_stop1500': SearchPolicy(0, (), None, 1500),
    'no_add_recovery1000_stop1500': SearchPolicy(1000, (0,), 1000, 1500),
    'partial_add_keep_exit': SearchPolicy(500, (5000,), None, 2000),
    'partial_add_break_even': SearchPolicy(500, (5000,), 0, 2000),
    'double_break_even_no_stop': SearchPolicy(500, (10000,), 0, None),
}


def main():
    coverage = json.loads((SOURCE / 'minute_coverage.json').read_text())
    eligible = {(r['ticker'], r['entry_date']) for r in coverage['coverage']}
    events = json.loads((SOURCE / 'minute_sample_five_year.json').read_text())
    results, provenance = [], []
    for event in events:
        if (event['ticker'], event['entry_date']) not in eligible:
            continue
        filename = SOURCE / 'source_data/minute_sample_five_year' / f"{event['ticker']}_{event['entry_date']}_{event['end_date_5']}.json.gz"
        with gzip.open(filename, 'rt') as handle:
            document = json.load(handle)
        assert document['adjusted'] is True
        sessions = defaultdict(list)
        for bar in sorted(document['bars'], key=lambda b: b['t']):
            when = local_time(bar)
            if 570 <= when.hour * 60 + when.minute < 960:
                sessions[when.date().isoformat()].append(bar)
        dates = sorted(sessions)
        assert dates[0] == event['entry_date'] and len(dates) == 5
        day = sessions[dates[0]]
        entry, close = day[0]['o'], day[-1]['c']
        held = close < entry
        exit_price = sessions[dates[1]][0]['o'] if held else close
        row = {'ticker': event['ticker'], 'date': dates[0], 'next_session': dates[1],
               'entry': entry, 'close': close, 'exit': exit_price, 'heldOvernight': held,
               'pnl': round(10000 * (exit_price / entry - 1), 2)}
        paths = [price_path(row, day, order) for order in ['OHLC', 'OLHC']]
        provenance.append({'ticker': event['ticker'], 'entry_date': dates[0],
                           'sha256': hashlib.sha256(filename.read_bytes()).hexdigest()})
        for bps in [0, 11, 26]:
            for name, policy in POLICIES.items():
                alternatives = sorted([replay(path, policy, bps / 10000) for path in paths], key=lambda r: r['pnl'])
                low, high = alternatives
                results.append({'ticker': event['ticker'], 'entry_date': dates[0], 'split': event['split'],
                                'policy': name, 'cost_bps_each_way': bps,
                                'budget': policy.budget,
                                'pnl_per_10k_reserved': round(low['pnl'] * 10000 / policy.budget, 4),
                                'upper_pnl_per_10k_reserved': round(high['pnl'] * 10000 / policy.budget, 4),
                                'added': low['adds'] > 0})
    assert len(provenance) == 433
    summaries = []
    for split in ['all', 'development', 'holdout']:
        for bps in [0, 11, 26]:
            base = {(r['ticker'], r['entry_date']): r['pnl_per_10k_reserved'] for r in results
                    if r['policy'] == 'original' and r['cost_bps_each_way'] == bps
                    and (split == 'all' or r['split'] == split)}
            for name in POLICIES:
                group = [r for r in results if r['policy'] == name and r['cost_bps_each_way'] == bps
                         and (split == 'all' or r['split'] == split)]
                values = sorted(r['pnl_per_10k_reserved'] for r in group)
                deltas = [r['pnl_per_10k_reserved'] - base[(r['ticker'], r['entry_date'])] for r in group]
                summaries.append({'split': split, 'cost_bps_each_way': bps, 'policy': name,
                                  'n': len(group), 'mean_per_10k_reserved': round(mean(values), 2),
                                  'mean_difference_vs_original': round(mean(deltas), 2),
                                  'worst': round(values[0], 2),
                                  'worst_5pct_mean': round(mean(values[:max(1, round(.05 * len(values)))]), 2),
                                  'win_pct': round(100 * sum(v > 0 for v in values) / len(values), 1),
                                  'positions_added': sum(r['added'] for r in group),
                                  'ambiguous_positions': sum(r['pnl_per_10k_reserved'] != r['upper_pnl_per_10k_reserved'] for r in group)})
    output = {
        'purpose': 'Secondary stress sample, not model selections and not an untouched validation set.',
        'dates': [min(r['entry_date'] for r in provenance), max(r['entry_date'] for r in provenance)],
        'policies': {name: asdict(p) for name, p in POLICIES.items()},
        'sampled': 480, 'retained': 433, 'existing_exclusion_reasons': coverage['exclusion_reasons'],
        'method': [
            'Signal: prior close-to-close drop of at least 5%, minimum price $5 and prior mean dollar volume $10M; 21-session ticker cooldown.',
            'Entry: first regular-session minute open on the following session. Exit: last regular-session minute close when up/flat, otherwise first regular minute open next session.',
            'Prices are split-adjusted minute aggregates throughout; cash dividends are not included. Prior five-session coverage and basis checks are reused unchanged.',
            'Fixed policy choices were made from the Rank sample before examining this follow-up output. The broad sample was already used for earlier research and is not pristine holdout data.',
            'The historical split label holdout means entries from 2025 onward; development means earlier entries. It does not imply the data remain unseen.',
            'Overlapping events across tickers are independent hypothetical positions for comparison, not a fundable sequential portfolio.',
            'Current-universe survivorship, aggregate-bar fills, possible closing-auction differences, spreads and market impact limit interpretation.',
            'The same capital reserve includes all planned additions and earns zero while uninvested; capital, stops and targets scale together.',
        ],
        'summaries': summaries, 'rows': results, 'source_hashes': provenance,
    }
    (ROOT / 'average_down_historical_stress.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'dates': output['dates'], 'summary': [s for s in summaries if s['cost_bps_each_way'] == 11 and s['split'] != 'development']}, indent=2))


if __name__ == '__main__':
    main()
