"""Fixed-policy daily-bar comparison; the intraday study is separate."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from simulation import POLICIES,simulate,summarize,clustered_mean_ci

ROOT=Path(__file__).resolve().parent
events=pd.read_json(ROOT/'events_five_year.json',convert_dates=False)
prices=pd.read_pickle(ROOT/'five_year_arrays.pkl')
ref=next(iter(prices.values()))['dates']
times=pd.to_datetime(ref).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30)
seconds=times.as_unit('s').asi8.astype(float)
results=[]
for n,e in enumerate(events.to_dict('records')):
    d=prices[e['ticker']];start=e['entry_idx'];bars=d['bars'][start:start+20]
    stable=bars[:,3]>d['bars'][start-1:start+19,1]
    for policy in POLICIES:
        r=simulate(bars,seconds[start:start+20],stable,policy,daily_vol=e['vol20'],bar_seconds=23400)
        results.append({'ticker':e['ticker'],'entry_date':e['entry_date'],'split':e['split'],'ticker_holdout':e['ticker_holdout'],'policy':policy.name,**r})
    if (n+1)%5000==0:print('daily events',n+1,flush=True)
f=pd.DataFrame(results);f.to_parquet(ROOT/'daily_results.parquet',index=False)
summaries=[]
for (split,policy),g in f.groupby(['split','policy']):summaries.append({'split':split,'policy':policy,**summarize(g)})
(ROOT/'daily_summary.json').write_text(json.dumps(summaries,indent=2))
outcomes=[]
for split,g in events.groupby('split'):
    for h in (1,2,5,10,20):
        x=g[f'return_{h}']
        outcomes.append({'split':split,'horizon':h,'n':len(g),'positive_close_pct':float((x>0).mean()*100),'mean_return_pct':float(x.mean()*100),'median_return_pct':float(x.median()*100),'p05_return_pct':float(x.quantile(.05)*100),'worst_return_pct':float(x.min()*100),'touched_pre_drop_close_pct':float(g[f'recover_pre_drop_close_{h}'].mean()*100),'further_5pct_fall_pct':float((g[f'min_return_{h}']<=-.05).mean()*100),'further_10pct_fall_pct':float((g[f'min_return_{h}']<=-.10).mean()*100),'touched_2pct_gain_pct':float((g[f'max_return_{h}']>=.02).mean()*100)})
(ROOT/'event_outcomes.json').write_text(json.dumps(outcomes,indent=2))
hold=f[f.split=='holdout'];base=hold[hold.policy=='single_10k'][['ticker','entry_date','pnl']].rename(columns={'pnl':'base'})
comparisons=[]
for policy in [p.name for p in POLICIES if p.name!='single_10k']:
    g=hold[hold.policy==policy].merge(base,on=['ticker','entry_date']);g['difference']=g.pnl-g.base
    comparisons.append({'policy':policy,'mean_difference_vs_single10k':float(g.difference.mean()),'ci':clustered_mean_ci(g,'difference',repetitions=1000)})
(ROOT/'daily_paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
subgroups=[]
for label,mask in [('above_ma50',events.above_ma50),('below_ma50',~events.above_ma50),('strong_close',events.close_location>=.67),('weak_close',events.close_location<=.33),('volume_ge_2x',events.relative_volume>=2),('volume_lt_2x',events.relative_volume<2),('drop_ge_10pct',events['drop']<=-.10),('drop_5_to_10pct',events['drop']>-.10),('low_vol_lt3pct',events.vol20<.03),('high_vol_ge3pct',events.vol20>=.03)]:
    g=events[(events.split=='holdout')&mask]
    subgroups.append({'group':label,'n':len(g),'positive_close_5_pct':float((g.return_5>0).mean()*100),'mean_5_pct':float(g.return_5.mean()*100),'median_5_pct':float(g.return_5.median()*100),'p05_5_pct':float(g.return_5.quantile(.05)*100),'further_10pct_within5_pct':float((g.min_return_5<=-.10).mean()*100)})
(ROOT/'subgroup_outcomes.json').write_text(json.dumps(subgroups,indent=2))
print(pd.DataFrame(summaries).query("split=='holdout'")[['policy','n','mean','median','win_pct','expected_shortfall_5','avg_max_capital']].to_string(index=False),flush=True)
print('done',len(f),flush=True)
