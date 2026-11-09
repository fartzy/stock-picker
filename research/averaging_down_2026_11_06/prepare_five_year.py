"""Prepare the five-year provider panel and frozen, nonoverlapping event sample."""
from pathlib import Path
import json,gzip,hashlib
from collections import Counter
import pandas as pd
import numpy as np
from simulation import POLICIES
from dataclasses import asdict

ROOT=Path(__file__).resolve().parent
source=ROOT/'source_data/daily_five_year'

def frame(path):
    with gzip.open(path,'rt') as f:doc=json.load(f)
    if not doc['bars']:return None
    d=pd.DataFrame(doc['bars'])
    d.index=pd.to_datetime(d.t,unit='ms',utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    assert not d.index.has_duplicates,path.name
    return d.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})[['Open','High','Low','Close','Volume']].sort_index()

spy=frame(source/'SPY.json.gz');reference=spy.index
prices={};events=[];exclusions=Counter();manifest=[]
for p in sorted(source.glob('*.json.gz')):
    ticker=p.name.removesuffix('.json.gz');d=frame(p)
    if d is None:exclusions['empty_source']+=1;continue
    d=d.reindex(reference);x=d.to_numpy(dtype=float);o,h,l,c,v=x.T
    valid=np.isfinite(x).all(axis=1)&(np.min(x[:,:4],axis=1)>0)&(v>0)&(h+1e-5>=np.maximum(o,c))&(l-1e-5<=np.minimum(o,c))&(h>=l)
    avg=pd.Series(c*v).shift(1).rolling(20).mean().to_numpy()
    ma=pd.Series(c).rolling(50).mean().to_numpy()
    ret=pd.Series(c).pct_change(fill_method=None).to_numpy()
    vol=pd.Series(ret).rolling(20).std().to_numpy()
    prices[ticker]={'dates':reference.astype(str).tolist(),'bars':x,'valid':valid,'returns':ret,'ma50':ma,'vol20':vol,'adv20':avg}
    manifest.append({'ticker':ticker,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'valid_bars':int(valid.sum()),'invalid_or_missing':int((~valid).sum())})
    if ticker in ('MRCY','SPY'):continue
    last=-100
    for i in range(60,len(d)-21):
        if not np.isfinite(ret[i]) or ret[i]>-.05:continue
        if i-last<=21:exclusions['cooldown']+=1;continue
        if not valid[i-60:i+22].all():exclusions['incomplete_or_invalid_window']+=1;continue
        if c[i]<5 or avg[i]<1e7:exclusions['price_or_liquidity']+=1;continue
        last=i
        e={'ticker':ticker,'signal_idx':i,'entry_idx':i+1,'signal_date':str(reference[i].date()),'entry_date':str(reference[i+1].date()),'end_date_5':str(reference[i+5].date()),'end_date_20':str(reference[i+20].date()),'drop':ret[i],'body_return':c[i]/o[i]-1,'close_location':(c[i]-l[i])/(h[i]-l[i]) if h[i]>l[i] else .5,'above_ma50':bool(c[i]>=ma[i]),'relative_volume':v[i]/np.mean(v[i-20:i]),'vol20':vol[i-1],'adv20':avg[i],'split':'holdout' if str(reference[i+1].date())>='2025-01-01' else 'development','ticker_holdout':int(hashlib.sha256(ticker.encode()).hexdigest()[:8],16)%5==0}
        for horizon in (1,2,5,10,20):
            end=i+horizon;entry=o[i+1]
            e[f'return_{horizon}']=c[end]/entry-1
            e[f'min_return_{horizon}']=float(np.min(l[i+1:end+1])/entry-1)
            e[f'max_return_{horizon}']=float(np.max(h[i+1:end+1])/entry-1)
            e[f'recover_pre_drop_close_{horizon}']=bool(np.max(h[i+1:end+1])>=c[i-1])
        events.append(e)
f=pd.DataFrame(events);f.to_json(ROOT/'events_five_year.json',orient='records',indent=2)
pd.to_pickle(prices,ROOT/'five_year_arrays.pkl')
(ROOT/'five_year_input_manifest.json').write_text(json.dumps(manifest,indent=2))
rng=np.random.default_rng(20261106);samples=[]
for split in ('development','holdout'):
    group=f[f['split']==split];chosen=rng.choice(group.index.to_numpy(),size=min(240,len(group)),replace=False)
    samples.extend(f.loc[chosen].to_dict('records'))
(ROOT/'minute_sample_five_year.json').write_text(json.dumps(sorted(samples,key=lambda e:(e['entry_date'],e['ticker'])),indent=2))
summary={'events':len(f),'tickers':int(f.ticker.nunique()),'entry_dates':int(f.entry_date.nunique()),'split_counts':f['split'].value_counts().to_dict(),'first_entry':f.entry_date.min(),'last_entry':f.entry_date.max(),'exclusions':dict(exclusions),'minute_sample':len(samples),'available_tickers':len(prices),'calendar_start':str(reference.min().date()),'calendar_end':str(reference.max().date()),'calendar_sessions':len(reference)}
(ROOT/'five_year_cohort_summary.json').write_text(json.dumps(summary,indent=2))
config=json.loads((ROOT/'study_config.json').read_text());config.update({'expanded_before_policy_evaluation':True,'daily_source':'Massive split-adjusted daily bars; cash dividends excluded','study_start':'2021-11-01','study_end':'2026-11-06','holdout_start':'2025-01-01','minute_validation_target_events':480,'policies':[asdict(p) for p in POLICIES]})
(ROOT/'study_config.json').write_text(json.dumps(config,indent=2))
print(json.dumps(summary))
