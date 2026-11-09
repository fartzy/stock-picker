"""Read-only price audit and reproducible event selection; never touches live data."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from collections import Counter
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
config=json.loads((ROOT/'study_config.json').read_text())
reg=pd.read_parquet(PROJECT/'data/universe/registry.parquet')
reference=pd.read_parquet(PROJECT/'data/prices/MRCY.parquet').index
prices={};exclusions=Counter();manifest=[];events=[]
for ticker in sorted(reg.index):
 p=PROJECT/'data/prices'/f'{ticker}.parquet'
 if not p.exists():exclusions['missing_file']+=1;continue
 d=pd.read_parquet(p).reindex(reference).copy()
 cols=['Open','High','Low','Close','Volume']
 x=d[cols].to_numpy(dtype=float);o,h,l,c,v=x.T
 valid=np.isfinite(x).all(axis=1)&(np.min(x[:,:4],axis=1)>0)&(v>0)&(h+1e-5>=np.maximum(o,c))&(l-1e-5<=np.minimum(o,c))&(h>=l)
 avg=pd.Series(c*v).shift(1).rolling(20).mean().to_numpy()
 ma=pd.Series(c).rolling(50).mean().to_numpy()
 ret=pd.Series(c).pct_change(fill_method=None).to_numpy()
 vol=pd.Series(ret).rolling(20).std().to_numpy()
 factor=(d['Adj Close']/d['Close']).to_numpy() if 'Adj Close' in d else np.ones(len(d))
 action=np.abs(pd.Series(factor).pct_change(fill_method=None).fillna(0).to_numpy())>.001
 prices[ticker]={'dates':reference.astype(str).tolist(),'bars':x,'valid':valid,'returns':ret,'ma50':ma,'vol20':vol,'adv20':avg}
 manifest.append({'ticker':ticker,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':int(valid.sum())})
 if ticker=='MRCY':continue
 last=-100
 for i in range(60,len(d)-21):
  if not np.isfinite(ret[i]) or ret[i]>-.05:continue
  if i-last<=21:exclusions['cooldown']+=1;continue
  if not valid[i-60:i+22].all():exclusions['incomplete_or_invalid_window']+=1;continue
  if action[i-20:i+22].any():exclusions['adjustment_factor_change']+=1;continue
  if c[i]<5 or avg[i]<1e7:exclusions['price_or_liquidity']+=1;continue
  last=i
  e={'ticker':ticker,'signal_idx':i,'entry_idx':i+1,'signal_date':str(reference[i].date()),'entry_date':str(reference[i+1].date()),'end_date_5':str(reference[i+5].date()),'end_date_20':str(reference[i+20].date()),'drop':ret[i],'body_return':c[i]/o[i]-1,'close_location':(c[i]-l[i])/(h[i]-l[i]) if h[i]>l[i] else .5,'above_ma50':bool(c[i]>=ma[i]),'relative_volume':v[i]/np.mean(v[i-20:i]),'vol20':vol[i-1],'adv20':avg[i],'split':'holdout' if str(reference[i+1].date())>='2026-07-01' else 'development'}
  for horizon in (1,2,5,10,20):
   end=i+horizon;entry=o[i+1]
   e[f'return_{horizon}']=c[end]/entry-1
   e[f'min_return_{horizon}']=float(np.min(l[i+1:end+1])/entry-1)
   e[f'max_return_{horizon}']=float(np.max(h[i+1:end+1])/entry-1)
   e[f'recover_pre_drop_close_{horizon}']=bool(np.max(h[i+1:end+1])>=c[i-1])
  events.append(e)
f=pd.DataFrame(events);f.to_json(ROOT/'events.json',orient='records',indent=2)
# Trusted local cache of arrays, input hashes retained in manifest.
pd.to_pickle(prices,ROOT/'daily_arrays.pkl')
(ROOT/'selected_price_manifest.json').write_text(json.dumps(manifest,indent=2))
rng=np.random.default_rng(config['minute_validation_seed']);samples=[]
for split in ('development','holdout'):
 group=f[f['split']==split];n=min(config['minute_validation_target_events']//2,len(group))
 chosen=rng.choice(group.index.to_numpy(),size=n,replace=False);samples.extend(f.loc[chosen].to_dict('records'))
(ROOT/'minute_sample.json').write_text(json.dumps(sorted(samples,key=lambda e:(e['entry_date'],e['ticker'])),indent=2))
summary={'events':len(f),'tickers':int(f.ticker.nunique()),'entry_dates':int(f.entry_date.nunique()),'split_counts':f['split'].value_counts().to_dict(),'first_entry':f.entry_date.min(),'last_entry':f.entry_date.max(),'exclusions':dict(exclusions),'minute_sample':len(samples),'available_tickers':len(prices)}
(ROOT/'cohort_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
