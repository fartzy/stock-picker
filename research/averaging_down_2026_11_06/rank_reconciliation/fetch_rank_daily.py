"""Independent daily open/close check for prices used by the saved paper book."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, API_ROOT

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / 'dashboard_rows.json').read_text())


def fetch(row):
    key = {'ticker': row['ticker'], 'date': row['date']}
    try:
        payload = MassiveOvernightClient()._get_payload(
            API_ROOT + f"/v1/open-close/{row['ticker']}/{row['date']}", {'adjusted': 'false'})
        return {**key, 'payload': payload}
    except Exception as error:
        return {**key, 'error': type(error).__name__}


results = list(ThreadPoolExecutor(max_workers=6).map(fetch, rows))
(ROOT / 'daily_price_snapshot.json').write_text(json.dumps(results, indent=2))
print('Daily summaries:', len(results), 'errors:', sum('error' in r for r in results))
for row, result in zip(rows, results):
    price = result.get('payload', {}).get('open')
    if price and abs(price / row['entry_price'] - 1) > .005:
        print(row['date'], row['ticker'], 'saved', row['entry_price'], 'verified daily', price)
