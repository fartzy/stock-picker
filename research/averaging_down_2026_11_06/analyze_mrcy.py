"""Separate retrospective case: true broker cash totals and timestamp caveats."""
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import json,gzip
import numpy as np
import pandas as pd
from simulation import POLICIES,simulate
from minute_data import regular_frame,stability_flags

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
orders=[]
for day in ('2026-11-04','2026-11-05','2026-11-06'):
    orders += [o for o in json.loads((PROJECT/f'data/trades/broker_orders_{day}.json').read_text())['orders'] if o['ticker']=='MRCY']
orders.sort(key=lambda o:o['executed_at'])
net=sum(Decimal(str(o['broker_net_amount']))*(1 if o['side']=='sell' else -1) for o in orders)
by_day=[];q=0;cash=Decimal(0)
for day in ('2026-11-04','2026-11-05','2026-11-06'):
    a=[o for o in orders if o['executed_at'].startswith(day)]
    bought=sum(o['shares'] for o in a if o['side']=='buy');sold=sum(o['shares'] for o in a if o['side']=='sell')
    cost=sum(Decimal(str(o['broker_net_amount'])) for o in a if o['side']=='buy');proceeds=sum(Decimal(str(o['broker_net_amount'])) for o in a if o['side']=='sell')
    q+=bought-sold;cash+=proceeds-cost
    by_day.append({'day':day,'bought':bought,'sold':sold,'cost':float(cost),'net_proceeds':float(proceeds),'ending_shares':q,'cumulative_cash':float(cash)})
frames=[]
for day in ('2026-11-04','2026-11-05','2026-11-06'):
    frames.append(regular_frame(json.loads((ROOT/f'source_data/MRCY_{day}_minute.json').read_text())))
d=pd.concat(frames);entry=pd.Timestamp('2026-11-04T09:36:02-05:00')
d=d[d.index>=entry.floor('min')];a=d[['o','h','l','c','v']].to_numpy(dtype=float)
times=d.index.as_unit('s').asi8.astype(float);stable=stability_flags(d)
results=[]
for p in POLICIES:
    results.append({'policy':p.name,**simulate(a,times,stable,p,entry_price=78.74,keep_path=True)})
# The Nov 4 intraday sequence includes provisional timestamps. From Nov 5 on,
# prior-day total cash and shares are exact and only one Nov 5 sale time is provisional.
q=900.;cash=by_day[0]['cumulative_cash'];peak_cash=-cash;peak_shares=q
following=[o for o in orders if o['executed_at']>='2026-11-05'];j=0;actual=[]
for stamp,b in d[d.index>=pd.Timestamp('2026-11-05T00:00:00-05:00')].iterrows():
    end=stamp+pd.Timedelta(minutes=1)
    while j<len(following) and pd.Timestamp(following[j]['executed_at'])<end:
        o=following[j];sign=1 if o['side']=='buy' else -1;q+=sign*o['shares'];cash-=sign*o['broker_net_amount'];j+=1
        peak_cash=max(peak_cash,-cash);peak_shares=max(peak_shares,q)
    actual.append({'time':stamp.timestamp()+60,'pnl':cash+q*b.c,'shares':q,'price':b.c,'net_cash_committed':-cash})
assert q==0 and abs(cash-float(net))<.01
late=[o for o in orders if o['side']=='sell' and o['executed_at']>='2026-11-05T15:50:00-05:00']
late_shares=sum(o['shares'] for o in late);late_proceeds=sum(o['broker_net_amount'] for o in late)
with gzip.open(ROOT/'source_data/daily_five_year/MRCY.json.gz','rt') as f:daily=json.load(f)
closing=daily['bars'][-1]['c']
counterfactual=float(net)+late_shares*closing-late_proceeds
levels=np.array([1,.95,.95**2,.95**3]);normalized_qty=np.sum(10000/levels)
average_entry = 40000 / normalized_qty
loss_at_minus20 = float(normalized_qty * .8 - 40000)
math = {
    'levels_pct': list((levels - 1) * 100),
    'average_entry_fraction': average_entry,
    'loss_at_fourth_fill': float(normalized_qty * levels[-1] - 40000),
    'loss_at_20pct_below_initial': loss_at_minus20,
    'loss_at_30pct_below_initial': float(normalized_qty * .7 - 40000),
    'bounce_needed_from_fourth_fill_pct': float((average_entry / levels[-1] - 1) * 100),
    'tranche_for_2k_loss_at_minus20pct_before_costs': 10000 * 2000 / -loss_at_minus20,
}
summary={'orders':len(orders),'provisional_timestamps':sum(bool(o.get('timestamp_is_provisional')) for o in orders),'realized_cash_pnl':float(net),'fees':float(sum(Decimal(str(o['manual_fee'])) for o in orders)),'by_day':by_day,'nov5_peak_net_cash_committed':peak_cash,'nov5_peak_shares':peak_shares,'worst_campaign_mark_from_nov5_minute_closes':min(x['pnl'] for x in actual),'nov6_open':daily['bars'][-1]['o'],'nov6_low':daily['bars'][-1]['l'],'nov6_high':daily['bars'][-1]['h'],'nov6_close':closing,'hindsight_late_sales_shares':late_shares,'hindsight_late_sales_proceeds':late_proceeds,'hindsight_hold_to_nov6_close_campaign_pnl_before_new_exit_costs':counterfactual,'ladder_arithmetic':math,'simulation_results':[{k:v for k,v in r.items() if k!='path'} for r in results]}
(ROOT/'mrcy_summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'mrcy_paths.json').write_text(json.dumps({'simulations':results,'actual_from_nov5':actual,'prices':[{'time':t.timestamp(),'price':r.c} for t,r in d.iterrows()]},separators=(',',':')))
print(json.dumps({k:v for k,v in summary.items() if k!='simulation_results'},indent=2))
print(pd.DataFrame(summary['simulation_results'])[['policy','pnl','max_capital','worst_mark','exit_reason']].to_string(index=False))
