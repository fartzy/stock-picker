"""Extend the existing fixed minute sample; download only sessions six onward."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gzip
import json
import time

import pandas as pd
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'source_data' / 'minute_two_week_extension'
OUT.mkdir(exist_ok=True)
events = json.loads((ROOT / 'minute_sample_five_year.json').read_text())
coverage = json.loads((ROOT / 'minute_coverage.json').read_text())
retained = {(e['ticker'], e['entry_date']) for e in coverage['coverage']}
daily = pd.read_pickle(ROOT / 'five_year_arrays.pkl')
requests = []
for event in events:
    if (event['ticker'], event['entry_date']) not in retained:
        continue
    dates = daily[event['ticker']]['dates'][event['entry_idx']:event['entry_idx'] + 10]
    cutoff = datetime.fromisoformat(event['entry_date']).date() + timedelta(days=14)
    dates = [d[:10] for d in dates if datetime.fromisoformat(d[:10]).date() <= cutoff]
    requests.append({**event, 'expected_dates': dates, 'extension_start': dates[5], 'end_date_10': dates[-1]})
(ROOT / 'two_week_sample.json').write_text(json.dumps(requests, indent=2))


def fetch(event):
    path = OUT / f"{event['ticker']}_{event['extension_start']}_{event['end_date_10']}.json.gz"
    if path.exists():
        return {'ticker': event['ticker'], 'entry_date': event['entry_date'], 'cached': True, 'ok': True}
    client = MassiveOvernightClient()
    endpoint = f"/v2/aggs/ticker/{event['ticker']}/range/1/minute/{event['extension_start']}/{event['end_date_10']}"
    try:
        bars = []
        for page in client._pages(endpoint, {'adjusted': 'true', 'sort': 'asc', 'limit': 50000}):
            bars.extend(page.get('results') or [])
        doc = {'ticker': event['ticker'], 'source': 'Massive', 'adjusted': True,
               'endpoint': endpoint, 'fetched_at': datetime.now(timezone.utc).isoformat(), 'bars': bars}
        with gzip.open(path, 'wt') as handle:
            json.dump(doc, handle, separators=(',', ':'))
        return {'ticker': event['ticker'], 'entry_date': event['entry_date'], 'bars': len(bars), 'ok': bool(bars)}
    except Exception as error:
        # Exception type only: provider URLs can contain credentials.
        return {'ticker': event['ticker'], 'entry_date': event['entry_date'], 'ok': False, 'error': type(error).__name__}


results = []
started = time.monotonic()
with ThreadPoolExecutor(max_workers=6) as pool:
    for future in as_completed([pool.submit(fetch, event) for event in requests]):
        results.append(future.result())
        if len(results) % 40 == 0:
            print(json.dumps({'completed': len(results), 'ok': sum(r['ok'] for r in results), 'seconds': round(time.monotonic() - started)}), flush=True)
(ROOT / 'two_week_fetch_manifest.json').write_text(json.dumps(results, indent=2))
print('Finished:', len(results), 'successful:', sum(r['ok'] for r in results), flush=True)
