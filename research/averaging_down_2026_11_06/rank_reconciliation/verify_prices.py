"""Separate input-price errors from arithmetic and execution assumptions."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from simulation import clustered_mean_ci

rows = json.loads((ROOT / 'dashboard_rows.json').read_text())
daily = {(r['date'], r['ticker']): r.get('payload', {})
         for r in json.loads((ROOT / 'daily_price_snapshot.json').read_text())}
out, mismatches = [], []
for row in rows:
    if not row['included_in_headline']:
        continue
    verified = daily[(row['date'], row['ticker'])]
    assert verified['status'] == 'OK' and verified['from'] == row['date'] and verified['symbol'] == row['ticker']
    opened, closed = verified['open'], verified['close']
    held = closed < opened
    exit_price = row['next_open'] if held else closed
    assert exit_price is not None
    result = {'date': row['date'], 'entry_date': row['date'], 'rank': row['rank'], 'ticker': row['ticker'],
              'saved_open': row['entry_price'], 'verified_open': opened,
              'saved_close': row['close_price'], 'verified_close': closed,
              'held_overnight_after_verification': held, 'exit_price': exit_price,
              'saved_pnl': row['display_total_pnl'], 'pnl': 10000 * (exit_price / opened - 1),
              'net11': 10000 * (exit_price / opened * .9989 / 1.0011 - 1),
              'net26': 10000 * (exit_price / opened * .9974 / 1.0026 - 1)}
    out.append(result)
    if abs(opened / row['entry_price'] - 1) > .005:
        history = pd.read_parquet(Path('data/prices') / f"{row['ticker']}.parquet")
        index = pd.to_datetime(history.index)
        prior = history[index < pd.Timestamp(row['date'])].iloc[-1]
        result['prior_session_open'] = float(prior.Open)
        result['saved_matches_prior_open'] = bool(abs(row['entry_price'] / prior.Open - 1) < .0001)
        mismatches.append(result)
f = pd.DataFrame(out)
source = pd.DataFrame([r for r in rows if r['included_in_headline']])
source['pnl'] = source.display_total_pnl
source['entry_date'] = source['date']
ordered = source.sort_values('date').groupby('date').pnl.sum()
curve = ordered.cumsum()
drawdown = np.maximum.accumulate(np.r_[0, curve.to_numpy()])[1:] - curve.to_numpy()


def stats(group, column='pnl'):
    x = group[column].to_numpy()
    return {'n': len(x), 'total_pnl': float(x.sum()), 'mean_per_position': float(x.mean()),
            'period_gain_on_20000': float(x.sum() / 20000), 'win_pct': float((x > 0).mean() * 100),
            'median': float(np.median(x)), 'worst': float(x.min()), 'best': float(x.max())}


summary = {
    'meaning': 'Same saved selections, verified daily open/close and the same down-day next-open rule. This is a price sensitivity, not a rerun of historical selection models.',
    'saved': stats(source), 'verified_gross': stats(f), 'verified_net11': stats(f, 'net11'),
    'verified_net26': stats(f, 'net26'),
    'verified_per_rank': [{'rank': int(rank), **stats(g), 'net11_total': float(g.net11.sum()),
                           'net11_mean': float(g.net11.mean())} for rank, g in f.groupby('rank')],
    'significant_open_mismatches': mismatches,
    'prior_open_matches': sum(r['saved_matches_prior_open'] for r in mismatches),
    'saved_cluster_bootstrap_95pct_mean': clustered_mean_ci(source),
    'verified_cluster_bootstrap_95pct_mean': clustered_mean_ci(f),
    'saved_closed_day_max_drawdown': float(drawdown.max()),
    'saved_top3_winners': float(source.pnl.nlargest(3).sum()),
    'saved_top5_winners': float(source.pnl.nlargest(5).sum()),
    'saved_without_top5': float(source.pnl.sum() - source.pnl.nlargest(5).sum()),
    'saved_worst_positions': source.sort_values('pnl')[['date', 'ticker', 'pnl']].head(5).to_dict('records'),
}
(ROOT / 'verified_price_rows.json').write_text(json.dumps(out, indent=2))
(ROOT / 'verified_price_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
