"""Read-only reconciliation: do not invent execution times or deduplicate fills."""
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
source = max(Path('/Users/michael.artz/Downloads').glob('Accounts_History*.csv'), key=lambda p: p.stat().st_mtime)
lines = source.read_text(encoding='utf-8-sig').splitlines()
header = next(i for i, line in enumerate(lines) if 'Run Date' in line and 'Symbol' in line)
reader = csv.DictReader(lines[header:])
records = []
for row in reader:
    if row.get('Symbol') == 'MRCY' and ('YOU BOUGHT' in row['Action'] or 'YOU SOLD' in row['Action']):
        records.append({'line': header + reader.line_num, 'date': datetime.strptime(row['Run Date'], '%m/%d/%Y').date().isoformat(),
                        'signed_shares': Decimal(row['Quantity']), 'amount': Decimal(row['Amount ($)'])})
orders = []
for day in ('2026-11-04', '2026-11-05', '2026-11-06'):
    orders.extend(o for o in json.loads((PROJECT / f'data/trades/broker_orders_{day}.json').read_text())['orders'] if o['ticker'] == 'MRCY')
csv_counts = Counter((r['date'], r['signed_shares'], r['amount']) for r in records)
ledger_counts = Counter((o['executed_at'][:10], (1 if o['side'] == 'buy' else -1) * Decimal(str(o['shares'])),
                        (-1 if o['side'] == 'buy' else 1) * Decimal(str(o['broker_net_amount']))) for o in orders)
by_day = []
for day in sorted({r['date'] for r in records}):
    rows = [r for r in records if r['date'] == day]
    by_day.append({'date': day, 'rows': len(rows), 'bought': float(sum(r['signed_shares'] for r in rows if r['signed_shares'] > 0)),
                   'sold': float(-sum(r['signed_shares'] for r in rows if r['signed_shares'] < 0)), 'net_cash': float(sum(r['amount'] for r in rows))})
audit = {'source_path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'header_line': header + 1,
         'columns': reader.fieldnames, 'has_execution_time_column': False,
         'only_time_text': [s for s in lines if re.search(r'\b\d{1,2}:\d{2}\b', s)],
         'mrcy_csv_rows': len(records), 'mrcy_ledger_orders': len(orders),
         'mrcy_csv_cash_pnl': float(sum(r['amount'] for r in records)), 'mrcy_net_shares': float(sum(r['signed_shares'] for r in records)),
         'by_day': by_day, 'csv_unmatched': [(str(k), v) for k, v in (csv_counts - ledger_counts).items()],
         'ledger_unmatched': [(str(k), v) for k, v in (ledger_counts - csv_counts).items()],
         'unresolved_provisional_timestamps': sum(bool(o.get('timestamp_is_provisional')) for o in orders),
         'note': 'CSV records one 200-share sale as 60 and 140 shares; their net proceeds total one cent less. No timestamps or ledger entries changed.'}
(ROOT / 'latest_csv_reconciliation.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
