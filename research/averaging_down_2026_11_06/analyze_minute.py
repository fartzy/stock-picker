"""Out-of-sample minute validation of fixed policies on a fixed event sample."""
from pathlib import Path
from collections import Counter
import json,gzip
import numpy as np
import pandas as pd
from simulation import POLICIES,simulate,summarize,clustered_mean_ci
from minute_data import regular_frame,stability_flags

ROOT=Path(__file__).resolve().parent
events=json.loads((ROOT/'minute_sample_five_year.json').read_text())
daily=pd.read_pickle(ROOT/'five_year_arrays.pkl')

results=[];excluded=[];coverage=[]
for n,e in enumerate(events):
    p=ROOT/'source_data/minute_sample_five_year'/f"{e['ticker']}_{e['entry_date']}_{e['end_date_5']}.json.gz"
    if not p.exists():excluded.append({**e,'reason':'missing_file'});continue
    with gzip.open(p,'rt') as f:d=regular_frame(json.load(f))
    if d.empty:excluded.append({**e,'reason':'empty'});continue
    expected=daily[e['ticker']]['dates'][e['entry_idx']:e['entry_idx']+5]
    expected=[x[:10] for x in expected]
    groups={str(day):g for day,g in d.groupby(d.index.date)}
    if sorted(groups)!=expected:excluded.append({**e,'reason':'missing_session'});continue
    failures=[]
    for day,g in groups.items():
        minute=g.index.hour*60+g.index.minute
        if len(g)<180 or minute.min()>575 or minute.max()<955:failures.append(day)
    if failures:excluded.append({**e,'reason':'insufficient_session_coverage','days':failures});continue
    a=d[['o','h','l','c','v']].to_numpy(dtype=float)
    if not (np.isfinite(a).all() and (a[:,:4]>0).all() and (a[:,1]+1e-5>=np.maximum(a[:,0],a[:,3])).all() and (a[:,2]-1e-5<=np.minimum(a[:,0],a[:,3])).all()):
        excluded.append({**e,'reason':'invalid_bar'});continue
    # Cross-check daily-vs-minute price basis on every session; do not silently mix split bases.
    daily_bars=daily[e['ticker']]['bars'][e['entry_idx']:e['entry_idx']+5]
    basis_errors=[]
    for k,day in enumerate(expected):
        g=groups[day]
        basis_errors.extend([abs(g.iloc[0].o/daily_bars[k,0]-1),abs(g.iloc[-1].c/daily_bars[k,3]-1)])
    if max(basis_errors)>.02:excluded.append({**e,'reason':'daily_minute_basis_mismatch','largest_difference':max(basis_errors)});continue
    coverage.append({'ticker':e['ticker'],'entry_date':e['entry_date'],'bars':len(d),'per_day':[len(g) for g in groups.values()],'largest_daily_minute_difference':max(basis_errors)})
    times=d.index.as_unit('s').asi8.astype(float);stable=stability_flags(d)
    for policy in POLICIES:
        r=simulate(a,times,stable,policy,daily_vol=e['vol20'])
        results.append({'ticker':e['ticker'],'entry_date':e['entry_date'],'split':e['split'],'policy':policy.name,'cost_bps':11,**r})
    # Sensitivity repeats execution decisions, not merely subtraction of a flat cost afterward.
    for cost_bps in (1,26):
        for policy in POLICIES:
            if policy.name not in ('single_10k','ladder_5','ladder_5_stop2k','ladder_5_stop2k_take2','ladder_stable_stop2k_take2'):continue
            r=simulate(a,times,stable,policy,daily_vol=e['vol20'],cost=cost_bps/10000)
            results.append({'ticker':e['ticker'],'entry_date':e['entry_date'],'split':e['split'],'policy':policy.name,'cost_bps':cost_bps,**r})
    if (n+1)%100==0:print('minute events examined',n+1,'retained',len(coverage),flush=True)
f=pd.DataFrame(results);f.to_parquet(ROOT/'minute_results.parquet',index=False)
summaries=[]
for (split,cost,policy),g in f.groupby(['split','cost_bps','policy']):summaries.append({'split':split,'cost_bps':int(cost),'policy':policy,**summarize(g)})
(ROOT/'minute_summary.json').write_text(json.dumps(summaries,indent=2))
(ROOT/'minute_coverage.json').write_text(json.dumps({'sampled':len(events),'retained':len(coverage),'excluded_count':len(excluded),'exclusion_reasons':dict(Counter(e['reason'] for e in excluded)),'coverage':coverage,'excluded':excluded},indent=2))
hold=f[(f.split=='holdout')&(f.cost_bps==11)];base=hold[hold.policy=='single_10k'][['ticker','entry_date','pnl']].rename(columns={'pnl':'base'})
comparisons=[]
for policy in [p.name for p in POLICIES if p.name!='single_10k']:
    g=hold[hold.policy==policy].merge(base,on=['ticker','entry_date']);g['difference']=g.pnl-g.base
    comparisons.append({'policy':policy,'mean_difference_vs_single10k':float(g.difference.mean()),'ci':clustered_mean_ci(g,'difference')})
(ROOT/'minute_paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
print(pd.DataFrame(summaries).query("split=='holdout' and cost_bps==11")[['policy','n','mean','median','win_pct','expected_shortfall_5','avg_max_capital']].to_string(index=False),flush=True)
print('coverage',len(coverage),dict(Counter(e['reason'] for e in excluded)),flush=True)

