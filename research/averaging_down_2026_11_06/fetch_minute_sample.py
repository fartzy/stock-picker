"""Bounded, reproducible historical read using the existing market-data account."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip,json,time
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'source_data/minute_sample_five_year';OUT.mkdir(exist_ok=True)
events=json.loads((ROOT/'minute_sample_five_year.json').read_text())
def fetch(e):
 path=OUT/f"{e['ticker']}_{e['entry_date']}_{e['end_date_5']}.json.gz"
 if path.exists():return {'ticker':e['ticker'],'date':e['entry_date'],'cached':True,'ok':True}
 client=MassiveOvernightClient();endpoint=f"/v2/aggs/ticker/{e['ticker']}/range/1/minute/{e['entry_date']}/{e['end_date_5']}"
 try:
  rows=[]
  for page in client._pages(endpoint,{'adjusted':'true','sort':'asc','limit':50000}):rows.extend(page.get('results') or [])
  doc={'ticker':e['ticker'],'start':e['entry_date'],'end':e['end_date_5'],'source':'Massive','adjusted':True,'endpoint':endpoint,'fetched_at':datetime.now(timezone.utc).isoformat(),'bars':rows}
  with gzip.open(path,'wt') as f:json.dump(doc,f,separators=(',',':'))
  return {'ticker':e['ticker'],'date':e['entry_date'],'bars':len(rows),'ok':bool(rows)}
 except Exception as ex:return {'ticker':e['ticker'],'date':e['entry_date'],'ok':False,'error':type(ex).__name__+': '+str(ex)[:100]}
results=[];start=time.monotonic()
with ThreadPoolExecutor(max_workers=6) as pool:
 futures=[pool.submit(fetch,e) for e in events]
 for f in as_completed(futures):
  results.append(f.result())
  if len(results)%20==0:print(json.dumps({'completed':len(results),'ok':sum(r['ok'] for r in results),'seconds':round(time.monotonic()-start,1)}),flush=True)
(ROOT/'minute_five_year_fetch_manifest.json').write_text(json.dumps(results,indent=2));print('finished',len(results),'successful',sum(r['ok'] for r in results),flush=True)
