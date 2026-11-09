"""Freeze unadjusted minute bars for the exact saved Rank Top 2 selections."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import gzip
import json

from stock_picker.ingestion.massive_overnight import MassiveOvernightClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'minute_bars'
OUT.mkdir(exist_ok=True)
ROWS = json.loads((ROOT / 'dashboard_rows.json').read_text())
AS_OF = date(2026, 11, 6)


def fetch(row):
    start = date.fromisoformat(row['date'])
    end = min(start + timedelta(days=13), AS_OF)
    path = OUT / f"{row['ticker']}_{start}_{end}.json.gz"
    key = {'ticker': row['ticker'], 'entry_date': str(start), 'end_date': str(end),
           'full_horizon': start + timedelta(days=13) <= AS_OF}
    if path.exists():
        return {**key, 'ok': True, 'cached': True}
    endpoint = f"/v2/aggs/ticker/{row['ticker']}/range/1/minute/{start}/{end}"
    try:
        bars = []
        for page in MassiveOvernightClient()._pages(endpoint, {'adjusted': 'false', 'sort': 'asc', 'limit': 50000}):
            bars.extend(page.get('results') or [])
        doc = {**key, 'source': 'Massive', 'adjusted': False, 'endpoint': endpoint,
               'retrieved_at': datetime.now(timezone.utc).isoformat(), 'bars': bars}
        with gzip.open(path, 'wt') as handle:
            json.dump(doc, handle, separators=(',', ':'))
        return {**key, 'ok': bool(bars), 'bars': len(bars)}
    except Exception as error:
        # Never serialize API exceptions or URLs containing credentials.
        return {**key, 'ok': False, 'error': type(error).__name__}


results = []
with ThreadPoolExecutor(max_workers=6) as pool:
    for future in as_completed([pool.submit(fetch, row) for row in ROWS]):
        results.append(future.result())
        if len(results) % 12 == 0:
            print('Fetched', len(results), 'of', len(ROWS), flush=True)
(ROOT / 'minute_fetch_manifest.json').write_text(json.dumps(results, indent=2))
print('Complete:', sum(r['ok'] for r in results), '/', len(results), flush=True)
