"""Freeze provider open/close prices for every saved completed-session Rank pick.

Uses the existing authenticated Massive client and retains a separate response
per ticker-date. Does not overwrite production prices, models, or paper books.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from stock_picker.ingestion.massive_overnight import API_ROOT, MassiveOvernightClient

ROOT = Path(__file__).resolve().parent


def fetch(row):
    client = MassiveOvernightClient()
    result = {key: row[key] for key in ('date', 'ticker')}
    provider_ticker = row['ticker'].replace('-', '.')
    result.update(source='Massive daily ticker summary', adjusted=False, provider_ticker=provider_ticker)
    for attempt in range(3):
        try:
            payload = client._get_payload(
                API_ROOT + f"/v1/open-close/{provider_ticker}/{row['date']}",
                {'adjusted': 'false'},
            )
            if payload.get('status') != 'OK':
                raise ValueError('Provider did not return OK')
            if payload.get('from') != row['date'] or payload.get('symbol') != provider_ticker:
                raise ValueError('Provider date or ticker mismatch')
            if not all(isinstance(payload.get(key), (int, float)) and
                       math.isfinite(payload[key]) and payload[key] > 0
                       for key in ('open', 'high', 'low', 'close')):
                raise ValueError('Missing or invalid OHLC price')
            if not payload['low'] <= min(payload['open'], payload['close']) <= max(payload['open'], payload['close']) <= payload['high']:
                raise ValueError('OHLC bounds do not reconcile')
            result.pop('error_type', None)
            result.update(payload=payload, fetched_at=datetime.now(timezone.utc).isoformat(), attempts=attempt+1)
            return result
        except Exception as error:
            # Never retain request URLs or credential-bearing exception strings.
            result['error_type'] = type(error).__name__
            if attempt < 2:
                time.sleep(2**attempt)
        finally:
            if attempt == 2 or 'payload' in result:
                client._session.close()
    result['fetched_at'] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    rows = [row for row in json.loads((ROOT/'input_rows.json').read_text()) if row['completed_session']]
    cache = ROOT/'prices'
    cache.mkdir(exist_ok=True)
    pending = []
    for row in rows:
        path = cache/f"{row['date']}_{row['ticker']}.json"
        if not path.exists() or 'payload' not in json.loads(path.read_text()):
            pending.append(row)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, row): row for row in pending}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = cache/f"{result['date']}_{result['ticker']}.json"
            path.write_text(json.dumps(result, indent=2)+'\n')
            if count % 60 == 0 or count == len(pending):
                print(f'Fetched {count}/{len(pending)} ticker-days', flush=True)
    responses = [json.loads((cache/f"{r['date']}_{r['ticker']}.json").read_text()) for r in rows]
    (ROOT/'price_snapshot.json').write_text(json.dumps(responses, indent=2)+'\n')
    errors = [r for r in responses if 'payload' not in r]
    print(json.dumps({'requested':len(rows), 'verified':len(rows)-len(errors), 'errors':errors}),flush=True)


if __name__ == '__main__':
    main()
