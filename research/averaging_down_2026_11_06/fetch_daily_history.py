"""Fetch a fixed five-year daily panel; preserves original application data."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip,json,time
import pandas as pd
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];OUT=ROOT/'source_data/daily_five_year';OUT.mkdir(exist_ok=True)
tickers=sorted(set(pd.read_parquet(PROJECT/'data/universe/registry.parquet').index)|{'SPY'})
def fetch(ticker):
 path=OUT/f'{ticker}.json.gz'
 if path.exists():
  with gzip.open(path,'rt') as f:d=json.load(f)
  return {'ticker':ticker,'bars':len(d['bars']),'ok':bool(d['bars']),'cached':True}
 client=MassiveOvernightClient();endpoint=f'/v2/aggs/ticker/{ticker}/range/1/day/2021-11-01/2026-11-06'
 try:
  rows=[]
  for page in client._pages(endpoint,{'adjusted':'true','sort':'asc','limit':50000}):rows.extend(page.get('results') or [])
  doc={'ticker':ticker,'source':'Massive','adjusted':True,'endpoint':endpoint,'fetched_at':datetime.now(timezone.utc).isoformat(),'bars':rows}
  with gzip.open(path,'wt') as f:json.dump(doc,f,separators=(',',':'))
  return {'ticker':ticker,'bars':len(rows),'ok':bool(rows)}
 except Exception as ex:return {'ticker':ticker,'ok':False,'error':type(ex).__name__+': '+str(ex)[:100]}
results=[];start=time.monotonic()
with ThreadPoolExecutor(max_workers=8) as pool:
 for f in as_completed([pool.submit(fetch,t) for t in tickers]):
  results.append(f.result())
  if len(results)%200==0:print(json.dumps({'completed':len(results),'ok':sum(r['ok'] for r in results),'seconds':round(time.monotonic()-start,1)}),flush=True)
(ROOT/'five_year_fetch_manifest.json').write_text(json.dumps(results,indent=2));print('finished',len(results),'successful',sum(r['ok'] for r in results),flush=True)
