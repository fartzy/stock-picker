"""Create standalone charts and a local HTML research report from saved results."""
import sys,os,json,html,base64
from pathlib import Path
sys.path.append('/private/tmp/stock-picker-research-plotting')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stock-picker-research-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
case=read('mrcy_summary.json');paths=read('mrcy_paths.json')
cohort=read('five_year_cohort_summary.json');coverage=read('minute_coverage.json')
daily=pd.DataFrame(read('daily_summary.json'));minute=pd.DataFrame(read('minute_summary.json'))
held=minute[(minute.split=='holdout')&(minute.cost_bps==11)].set_index('policy')
dh=daily[daily.split=='holdout'].set_index('policy')
labels={
 'single_10k':'$10K, no additions', 'full_40k':'$40K upfront',
 'ladder_5':'Four $10K purchases, 5% steps', 'ladder_5_stop2k':'5% additions + $2K loss trigger',
 'ladder_5_stop2k_be':'5% additions + $2K trigger + break-even exit',
 'ladder_5_stop2k_take2':'5% additions + $2K trigger + 2% gain exit',
 'ladder_5_stop2k_take4':'5% additions + $2K trigger + 4% gain exit',
 'ladder_5_stop1k_take2':'5% additions + $1K trigger + 2% gain exit',
 'ladder_5_stop4k_take2':'5% additions + $4K trigger + 2% gain exit',
 'ladder_stable_stop2k_take2':'Wait for rebound before adding; $2K / 2%',
 'single_stop500_take2':'$10K only; $500 loss trigger / 2% gain exit',
 'ladder_3_stop2k_take2':'3% addition steps; $2K / 2%',
 'ladder_7p5_stop2k_take2':'7.5% addition steps; $2K / 2%',
 'ladder_vol_stop2k_take2':'Volatility-scaled addition steps; $2K / 2%',
 'ladder_partial2_trail2':'Sell half at 2% gain, trail rest by 2%; $2K trigger',
}
palette=['#2563eb','#0d9488','#d97706','#7c3aed','#db2777']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cbd5e1','axes.labelcolor':'#334155','text.color':'#0f172a','xtick.color':'#475569','ytick.color':'#475569','axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'white','grid.alpha':.25,'savefig.facecolor':'white'})
dollars=FuncFormatter(lambda x,pos:f'-${abs(x):,.0f}' if x<0 else f'${x:,.0f}')
def save(fig,name):
    fig.savefig(ROOT/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(ROOT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

# Minute-by-minute MRCY replay, compressed overnight gaps, real cash path starts Nov 5.
times=np.array([x['time'] for x in paths['prices']]);prices=np.array([x['price'] for x in paths['prices']]);idx=np.arange(len(times))
dates=pd.to_datetime(times,unit='s',utc=True).tz_convert('America/New_York')
starts=[0]+[i for i in range(1,len(dates)) if dates[i].date()!=dates[i-1].date()]
fig,(ax,ay)=plt.subplots(2,1,figsize=(12,7.3),sharex=True,gridspec_kw={'height_ratios':[1,1.2]})
ax.plot(idx,prices,color='#1d4ed8',lw=1.7)
ax.axhline(78.74,color='#94a3b8',ls='--',lw=1,label='First recorded entry: $78.74')
ax.set_ylabel('MRCY price');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'${x:.0f}'));ax.grid(axis='y');ax.legend(loc='upper right',frameon=False)
ax.set_title('MRCY: the rebound was real, and so was the drawdown',loc='left',pad=15)
for name,color in [('single_10k',palette[0]),('ladder_5',palette[1]),('ladder_5_stop2k',palette[2])]:
    s=next(r for r in paths['simulations'] if r['policy']==name);p=s['path'];xx=np.searchsorted(times,[r['time']-1 for r in p]).clip(0,len(times)-1);yy=[r['pnl'] for r in p]
    if xx[-1]<idx[-1]:xx=np.r_[xx,idx[-1]];yy=yy+[yy[-1]]
    ay.plot(xx,yy,label=labels[name],lw=1.8,color=color)
actual=paths['actual_from_nov5'];xx=np.searchsorted(times,[r['time']-1 for r in actual]).clip(0,len(times)-1)
ay.plot(xx,[r['pnl'] for r in actual],label='Actual sequence (~$187K peak net cash committed)',color='#dc2626',lw=1.8)
ay.axhline(0,color='#64748b',lw=.8);ay.set_ylabel('P&L, dollars');ay.yaxis.set_major_formatter(dollars);ay.grid(axis='y')
ay.legend(loc='lower left',frameon=False,fontsize=9)
for a in (ax,ay):
    for i in starts[1:]:a.axvline(i,color='#cbd5e1',ls=':',lw=1)
ay.set_xticks(starts);ay.set_xticklabels([dates[i].strftime('%b %d · %I:%M %p ET') for i in starts]);ay.set_xlabel('Regular-session minutes; overnight gaps compressed')
fig.text(.01,.005,'Hypothetical strategies start at the first recorded entry price and include 11 bps per side. Actual Nov 4 intraday timing is incomplete.\nThe actual P&L line starts Nov 5 and excludes financing; its earlier cash balance is retained. Strategy exits use the last regular-session minute.',fontsize=8,color='#64748b')
fig.tight_layout(rect=[0,.055,1,1]);save(fig,'mrcy_replay')

chosen=['single_10k','ladder_5','ladder_5_stop2k','ladder_5_stop2k_take2']
short=['$10K, no adds','5% adds, no loss trigger','5% adds + $2K trigger','5% adds + $2K / 2% exit']
fig,(ax,ay)=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1,1]})
y=np.arange(len(chosen));colors=palette[:4]
ax.barh(y,[held.loc[k,'mean'] for k in chosen],color=colors,height=.6);ax.set_yticks(y,short);ax.invert_yaxis();ax.set_title('Average result per trial');ax.xaxis.set_major_formatter(dollars);ax.grid(axis='x')
ay.barh(y,[-held.loc[k,'expected_shortfall_5'] for k in chosen],color=colors,height=.6);ay.set_yticks(y,['']*4);ay.invert_yaxis();ay.set_title('Average loss in the worst 5%');ay.xaxis.set_major_formatter(dollars);ay.grid(axis='x')
fig.suptitle('More winning trades can come with much larger losses',x=.01,ha='left',fontweight='bold',fontsize=14)
fig.text(.01,.005,'217 later-period trials, 5-session maximum holding period, minute-close decisions / next-minute-open execution.\n11 bps per side; financing excluded. These are per-trial outcomes, not portfolio returns. Different policies deploy different amounts.',fontsize=8,color='#64748b')
fig.tight_layout(rect=[0,.09,1,.96]);save(fig,'risk_return_comparison')

comparisons={r['policy']:r for r in read('minute_paired_comparisons.json')};chosen_ci=chosen[1:]+['ladder_stable_stop2k_take2']
fig,ax=plt.subplots(figsize=(11,4.4));vals=[comparisons[k]['mean_difference_vs_single10k'] for k in chosen_ci];bounds=np.array([comparisons[k]['ci']['envelope'] for k in chosen_ci]);yy=np.arange(len(vals))
ax.errorbar(vals,yy,xerr=np.array([np.array(vals)-bounds[:,0],bounds[:,1]-np.array(vals)]),fmt='o',color='#2563eb',capsize=5,lw=2)
ax.axvline(0,color='#dc2626',ls='--');ax.set_yticks(yy,[labels[k] for k in chosen_ci]);ax.invert_yaxis();ax.set_xlabel('Difference in average P&L versus $10K with no additions');ax.xaxis.set_major_formatter(dollars);ax.grid(axis='x');ax.set_title('The minute sample does not establish a dependable advantage from adding',loc='left',pad=16)
fig.text(.01,.005,'Bars are the wider envelope of 95% ticker-cluster and entry-date-cluster bootstrap intervals.\nThis is an uncertainty diagnostic, not a formal multi-way confidence guarantee; all intervals span zero.',fontsize=8,color='#64748b')
fig.tight_layout(rect=[0,.08,1,1]);save(fig,'uncertainty')

o=pd.DataFrame(read('event_outcomes.json'));o=o[o.split=='holdout']
fig,ax=plt.subplots(figsize=(10.5,4.5));ax.plot(o.horizon,o.touched_pre_drop_close_pct,'o-',label='Touched the close from before the initial down day',color='#0d9488',lw=2)
ax.plot(o.horizon,o.further_10pct_fall_pct,'o-',label='Fell another 10% below the next-day entry',color='#dc2626',lw=2)
ax.set_xticks([1,2,5,10,20]);ax.set_xlabel('Trading sessions after entry');ax.set_ylabel('Share of historical events');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.0f}%'));ax.set_ylim(0,75);ax.grid(axis='y');ax.legend(frameon=False,loc='upper left');ax.set_title('Waiting longer increases both recovery opportunities and exposure to further falls',loc='left',pad=15,fontsize=12)
fig.text(.01,.005,'11,068 later-period events. The two outcomes can occur in the same trial, in either order.\nA touched recovery level is a hindsight observation, not evidence that a live order would have filled.',fontsize=8,color='#64748b')
fig.tight_layout(rect=[0,.07,1,1]);save(fig,'recovery_and_risk')

fall=np.linspace(0,.4,401);levels=np.array([1,.95,.95**2,.95**3]);loss=[]
for drop in fall:
    price=1-drop;active=levels[levels>=price-1e-12];loss.append(np.sum(10000*(price/active-1)))
fig,ax=plt.subplots(figsize=(10.5,4.4));ax.plot(fall*100,loss,color='#7c3aed',lw=2.5,label='Up to four fixed $10K purchases');ax.plot(fall*100,-10000*fall,color='#2563eb',lw=1.8,label='$10K only')
ax.axhline(-2000,color='#dc2626',ls='--',label='$2K loss budget');ax.scatter([14.2625],[-2901.25],color='#7c3aed',zorder=5);ax.annotate('Fourth purchase fills:\nalready down $2,901',(14.2625,-2901.25),xytext=(18,-1400),arrowprops={'arrowstyle':'->','color':'#64748b'},fontsize=9)
ax.set_xlabel('Decline from the initial entry');ax.set_ylabel('P&L before costs');ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.0f}%'));ax.yaxis.set_major_formatter(dollars);ax.grid(axis='y');ax.legend(frameon=False,loc='lower left');ax.set_title('A $40K position cap can still lose thousands',loc='left',pad=15)
fig.tight_layout();save(fig,'ladder_risk_math')

def money(x):return f'−${abs(x):,.2f}' if x<0 else f'${x:,.2f}'
def table(rows,columns):
    head=''.join(f'<th>{html.escape(label)}</th>' for key,label,fmt in columns)
    body=''.join('<tr>'+''.join(f'<td>{html.escape(fmt(r[key]))}</td>' for key,label,fmt in columns)+'</tr>' for r in rows)
    return '<div class="table-wrap"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'
def policy_table(frame,names=None):
    rows=[]
    for key,row in frame.iterrows():
        if names is not None and key not in names:continue
        rows.append({'name':labels[key],**row.to_dict()})
    return table(rows,[('name','Rule',str),('n','Trials',lambda x:f'{int(x):,}'),('mean','Mean P&L',money),('win_pct','Win rate',lambda x:f'{x:.1f}%'),('expected_shortfall_5','Worst 5% average',money),('worst','Worst trial',money),('avg_max_capital','Mean maximum dollars deployed',money)])
case_rows=[{'name':labels[r['policy']],**r} for r in case['simulation_results'] if r['policy'] in chosen]
case_table=table(case_rows,[('name','Hypothetical rule',str),('pnl','Final P&L',money),('max_capital','Maximum purchases',money),('worst_mark','Worst marked P&L',money),('exit_reason','Exit',str)])
costs=minute[(minute.split=='holdout')&(minute.cost_bps==26)].set_index('policy')
sources=[('Massive historical aggregate documentation','https://massive.com/docs/rest/stocks/aggregates/custom-bars'),('FINRA: stop orders in volatile markets','https://www.finra.org/investors/insights/stop-orders-factors-consider-during-volatile-markets'),('FINRA: concentration risk','https://www.finra.org/investors/insights/concentration-risk'),('FINRA: margin accounts','https://www.finra.org/rules-guidance/key-topics/margin-accounts')]
report=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MRCY and averaging down — historical risk study</title>
<style>
:root{{--ink:#14213d;--muted:#596579;--line:#dbe2ea;--accent:#2156ba;--bg:#f4f6fa;--paper:#fff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.65 system-ui,-apple-system,sans-serif}}main{{max-width:1160px;margin:auto;padding:56px 36px 80px}}header{{border-bottom:1px solid var(--line);padding-bottom:28px}}h1{{font-size:38px;line-height:1.15;letter-spacing:-1px;margin:12px 0 20px}}h2{{font-size:26px;margin:48px 0 14px}}h3{{font-size:20px;margin:28px 0 12px}}p{{max-width:940px}}.eyebrow{{font-size:12px;font-weight:750;letter-spacing:1.8px;color:var(--accent);text-transform:uppercase}}.lead{{font-size:21px;line-height:1.5;max-width:950px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}}.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px}}.card strong{{display:block;font-size:25px}}.card span,.note{{font-size:14px;color:var(--muted)}}.finding{{border-left:4px solid var(--accent);padding:15px 24px;background:#eaf0fc;margin:24px 0}}img{{width:100%;height:auto;background:white;border:1px solid var(--line);border-radius:10px;margin:16px 0 8px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{padding:13px 14px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left;white-space:normal;min-width:240px}}th{{background:#eaf0f7;line-height:1.4}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:9px;margin:20px 0}}a{{color:var(--accent)}}li{{margin:8px 0}}details{{background:white;border:1px solid var(--line);padding:16px 22px;margin:18px 0;border-radius:10px}}summary{{cursor:pointer;font-weight:650}}code{{font-size:13px;background:#edf1f7;padding:2px 5px;border-radius:3px}}footer{{margin-top:48px;padding-top:24px;border-top:1px solid var(--line)}}@media(max-width:700px){{main{{padding:28px 18px}}h1{{font-size:30px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:white;font-size:11pt}}main{{padding:0}}details{{display:block}}img,table{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Historical risk study · November 6, 2026</div><h1>When does adding to a losing position earn its risk?</h1><p class="lead">The strongest finding is about exposure: fixed additions can improve the win rate while concentrating much larger losses in the failures. This study does not establish a dependable, fast-recovery trading edge from automatically adding every 5%.</p></header>
<div class="cards"><div class="card"><strong>5 years</strong><span>Daily data requested, Nov 2021–Nov 2026</span></div><div class="card"><strong>{cohort['events']:,}</strong><span>Eligible events across {cohort['tickers']:,} stocks</span></div><div class="card"><strong>11,068</strong><span>Later-period daily evaluation events</span></div><div class="card"><strong>433</strong><span>Minute paths retained; 217 in later evaluation</span></div></div>
<p>The question is not whether MRCY rebounded: it did. The question is whether a repeatable rule could capture recoveries while controlling the losses when they fail. All figures below come from saved broker records or historical provider bars, with execution assumptions stated explicitly.</p>
<h2>1. What happened in MRCY</h2>
<p>The recorded November 4–6 sequence ended with <strong>{money(case['realized_cash_pnl'])}</strong> in cash P&L after broker-reported trade charges, before financing and tax effects. That is the complete MRCY sequence, not a claim about one day’s total account P&L. The ledger contains 93 MRCY entries, including 17 with provisional timestamps. November 4 intraday ordering therefore cannot be reconstructed exactly.</p>
<p>From November 5 onward, the holdings reached <strong>2,700 shares</strong>. Net cash committed to the entire idea, including earlier realized losses, peaked at <strong>{money(case['nov5_peak_net_cash_committed'])}</strong>. A $1 price move at 2,700 shares changes mark-to-market value by $2,700.</p>
<p>On November 6, MRCY opened at $66.93, fell to $64.3706, reached $73.91, and closed at $73.55. The next-day bounce was substantial, but it followed another low. The original $78.74 entry price was still above that day’s close; the benefit of holding depended on the accumulated lower-price purchases.</p>
<img src="mrcy_replay.png" alt="MRCY minute prices and actual versus hypothetical P&L paths">
<p>A hindsight calculation that retains the final 1,600 shares sold late November 5 and on November 6, then liquidates at November 6’s official close, changes the campaign to about <strong>{money(case['hindsight_hold_to_nov6_close_campaign_pnl_before_new_exit_costs'])}</strong> before the replacement exit’s trading costs and financing. This assumes the account could hold the shares, makes no prediction, and uses an exit selected after seeing the outcome.</p>
<h3>Smaller hypothetical plans, starting at the first actual entry</h3>{case_table}
<p class="note">These are hypothetical $10K-based strategies, not a rescaling of every actual purchase. The first fill is anchored to $78.74. Subsequent decisions use completed minute bars; executions occur at the next observed minute open. The replay ends at the last regular-session minute close ($73.57), which differs slightly from the official daily close ($73.55). This retrospective endpoint was selected after the actual rebound; the broad tests below use fixed five- and twenty-session horizons. Base modeled cost is 11 bps each way. “Worst marked P&L” includes minute lows and is not necessarily a realized loss.</p>
<div class="finding"><strong>What this case teaches:</strong> the $40K no-stop plan eventually earned about $482 but first endured about $4,580 of losses. The $2K-trigger plan exited before that recovery. If that adverse excursion is intolerable, the remedy is smaller exposure or a different strategy—not assuming the recovery will arrive before the account or investor gives up.</div>
<h2>2. What the broader market history says</h2>
<p>The broad cohort consists of liquid stocks closing at least 5% below the prior close. The initial hypothetical purchase is at the following session’s open. This is a <strong>post-drop entry study</strong>, not a backtest of the app’s morning selection model. MRCY was excluded from the broad comparisons. Events for the same ticker are separated by more than 21 sessions.</p>
<p>The daily panel covers 2021-11-01 through 2026-11-06; sufficient lookback and forward coverage produce entries from {cohort['first_entry']} through {cohort['last_entry']}. Development entries precede 2025; later evaluation entries are in 2025–2026. Entry eligibility uses a $5 minimum split-adjusted price, prior 20-session average dollar volume of at least $10M, and 60 prior sessions. Full future windows are required, a selection limitation discussed below.</p>
<img src="recovery_and_risk.png" alt="Historical recovery frequency and risk of a further ten percent drop across holding periods">
<p>Among later-period events, about <strong>29.9%</strong> touched the pre-drop close within five sessions, while <strong>15.9%</strong> fell another 10% below the next-day entry during that period. By 20 sessions those figures were <strong>57.2%</strong> and <strong>40.2%</strong>. A path can do both. More time mechanically creates more opportunities to touch a recovery price and also exposes the position to additional adverse moves.</p>
<p>For the 20-session daily decision version, adding increased mean P&L from roughly <strong>$272 to $452</strong>, and the win rate from 54.3% to 63.8%. However, average maximum capital deployed rose from $10,000 to $19,538, and the average loss in the worst 5% of trials rose from roughly <strong>$2,976 to $8,024</strong>. This is extra expected dollars accompanied by substantially greater capital use and loss concentration; it is not a free improvement.</p>
<p>The matched-date SPY comparison also matters: the stocks’ average 20-session price return was 2.94%, versus 1.66% for SPY over the same weighted windows. Only about 49.9% of stocks beat SPY, and the median excess return was approximately zero. These are raw, unadjusted-for-risk comparisons, not proof of stock-selection skill.</p>
<h2>3. The minute-based test: higher win rate, thin average gains</h2>
<p>A fixed random sample of 480 event windows was chosen before inspecting policy outcomes. All were fetched. Coverage checks retained 433 windows: 216 development and 217 later-period evaluation windows. Forty-six lacked the required session coverage and one failed a daily-versus-minute price-basis check. All policies below face the same retained later-period paths and a five-session maximum holding period.</p>
{policy_table(held,chosen)}
<img src="risk_return_comparison.png" alt="Average trial profits and average loss in the worst five percent of minute trials">
<p>The 5% additions / $2K trigger / 2% gain-exit rule won 82.0% of trials, yet earned only $24 on average. Its average winner was about $275 and its average loser about $1,123. With that payoff ratio, approximately 80.3% winners are needed merely to break even before financing. A high win rate can coexist with a fragile result.</p>
<img src="uncertainty.png" alt="Paired strategy improvement confidence intervals all span zero">
<p>The plain adding rule’s average advantage over the $10K baseline was about $46, but the uncertainty interval ranged roughly from −$104 to +$182 per trial. The capped quick-profit rule’s advantage was about −$14, with an interval roughly from −$147 to +$106. These ranges do not establish a dependable minute-level advantage. They also do not prove that all dip-buying strategies fail.</p>
<p>The tested short rebound confirmation, 3% versus 5% versus 7.5% spacing, volatility-scaled spacing, break-even exits, and partial-profit/trailing exits did not produce a clearly established minute-level improvement over the no-add baseline. The moving-average and volume subgroups are descriptive and were not validated as reliable decision signals.</p>
<details><summary>All 15 minute-policy results, base costs</summary>{policy_table(held)}</details>
<details><summary>All 15 daily-policy results, 20-session maximum</summary>{policy_table(dh)}</details>
<h3>Cost and execution sensitivity</h3>
<p>Base assumptions are 10 bps slippage plus 1 bp fees per side, fractional shares, and complete fills at the next available minute open. These are modeling choices, not claims about your actual broker charges. At 25 bps slippage plus 1 bp fees, the quick-profit rule’s average falls to roughly $7; the rebound-confirmation rule’s average becomes about −$5. If the purchases were fully financed at a hypothetical 12% annual rate, those become approximately −$2 and −$15. Your actual borrowing rate and borrowed amount were not assumed to equal these examples.</p>
{policy_table(costs)}
<p>Some simulated orders represent more than 10% of the execution minute’s recorded volume. The engine reports that participation but does not model queue position, partial fills, trading halts, or nonlinear market impact. Price bars alone cannot establish execution quality for a large order.</p>
<p class="note">As an execution diagnostic, excluding any path where one of the five core policies exceeds 10% participation leaves 136 common evaluation events. The quick-profit rule averages about −$7, compared with about +$73 for the no-add baseline. This is a post-execution selection check, not an unbiased entry filter; it further cautions against treating the full-sample $24 mean as a dependable edge.</p>
<h2>4. The arithmetic of your proposed $10K additions</h2>
<p>Four equal $10K purchases after successive 5% price declines occur at 100%, 95%, 90.25%, and 85.7375% of the first price. They produce an average entry about 92.4425% of the original price. Lowering the average cost does not remove the loss already incurred.</p>
<img src="ladder_risk_math.png" alt="Loss curve for fixed additions versus a single purchase and a two thousand dollar risk budget">
<ul><li>At the fourth fill, the position is already down <strong>$2,901</strong>, before costs.</li><li>From that fourth fill, price must rebound about <strong>7.82%</strong> to reach the aggregate purchase cost, before costs.</li><li>At 20% below the initial price, the loss is about <strong>$5,384</strong>.</li><li>At 30% below the initial price, the loss is about <strong>$9,711</strong>.</li></ul>
<p>For an illustrative $2K loss budget and a planned exit 20% below the first entry, the same four-entry shape would allow approximately <strong>$3,715 per purchase, or $14,859 total</strong>, before fees, slippage, and gap risk. This is sizing arithmetic, not a recommendation to use a 20% stop. No stop guarantees the assumed exit price.</p>
<p>A general sizing check is <code>Σ shares_i × (entry_i − planned_exit_price) + estimated costs ≤ idea_loss_budget</code>. Include the risk from every planned addition before the first purchase. Literal doubling of each new purchase would instead total $10K + $20K + $40K + $80K = $150K; the tested proposal uses equal additions.</p>
<h2>5. A decision framework supported by this study</h2>
<ol>
<li><strong>Set the dollar loss budget before entering, then derive position size.</strong> Account equity, borrowed exposure, other positions, and the maximum tolerable loss are still needed to choose personal dollar limits. A position cap is useful, but it is not itself a loss cap.</li>
<li><strong>Treat each addition as a fresh investment decision.</strong> A lower average cost is an accounting consequence, not evidence of positive expected returns. The specific price-only rules tested here do not justify automatically adding at every 5% decline.</li>
<li><strong>Charge all partial exits and renewed purchases to one idea-level budget.</strong> Do not reset the loss allowance after realizing a loss or reducing the displayed average price. The simulation has no re-entry after a full exit, so repeated attempts could lose more than these results.</li>
<li><strong>Make the holding period explicit.</strong> The current app’s morning forecasts target the open-to-close session. Holding overnight is a separate strategy with gap risk, financing, and event exposure. The historical five- and twenty-session tests are distinct alternatives, not permission to extend a losing trade indefinitely.</li>
<li><strong>Choose an exit policy while flat, and assess its full payoff distribution.</strong> Quick recovery targets increase win rate but cap many gains; partial-profit and trailing exits were not a clear fix in this sample. A missed rebound does not invalidate a risk limit chosen for survivability.</li>
<li><strong>Use paper trading to validate the next version.</strong> Preserve a no-add baseline, predeclare the risk budget and time exit, and measure costs and fill quality. The present evidence supports exposure discipline more strongly than a particular dip-buying threshold.</li>
</ol>
<p>Prior trading profits can inform how much account drawdown you are willing to tolerate. They provide no new information about a stock’s recovery probability. If approaching the loss limit causes the planned exit to be postponed, the limit ceases to perform its intended role.</p>
<p>Stop orders can fill materially beyond their trigger price; stop-limit orders can fail to fill. The minute simulation includes gap and delayed-execution overshoots—for example, its $2K-trigger variants include losses above $3K. <a href="{sources[1][1]}">FINRA’s stop-order explanation</a> describes these execution limitations. A margin deficiency can also lead to broker liquidation, so an investor may not control how long a leveraged position remains open. See <a href="{sources[3][1]}">FINRA’s margin guidance</a>. Concentration across several correlated positions can make the account’s total risk larger than a per-stock rule suggests; see <a href="{sources[2][1]}">FINRA’s concentration-risk explanation</a>.</p>
<h2>6. What remains uncertain</h2>
<ul><li><strong>Survivorship and selection:</strong> the universe is based on recently selected stocks, not a complete point-in-time universe including every failed or delisted company. Requiring complete future windows can further understate failure risk. The observed frequencies are conditional on this dataset.</li><li><strong>Corporate actions:</strong> the broad data are provider split-adjusted price bars. Cash dividends, tax effects, point-in-time symbol identity changes, and independently verified corporate-action records are not fully modeled. These are exploratory results, not the repository’s certified overnight-data contract.</li><li><strong>Chronological holdout, not a live forward test:</strong> the later period was kept separate in reporting; it is still historical. Fifteen predeclared policies and descriptive subgroup comparisons do not eliminate multiple-comparison risk.</li><li><strong>Dependence:</strong> stocks share market shocks and recur across dates. Bootstrap diagnostics cluster by ticker and entry date separately, taking the wider interval; they are not a formal multi-way coverage guarantee.</li><li><strong>Portfolio feasibility:</strong> per-event profits cannot be summed into an investable portfolio return. Concurrent opportunities, capital reservations, correlated exposure, and margin constraints were not simulated as a full account.</li><li><strong>Event information:</strong> earnings, guidance changes, news, fundamentals, and point-in-time news classifications are not validated predictors in this study. A falling price alone cannot distinguish temporary selling pressure from permanent impairment.</li><li><strong>Actual records:</strong> 17 MRCY entries retain provisional times; full-period cash P&L is reconstructable, but the November 4 intraday path is not exact.</li></ul>
<h2>Reproducibility and files</h2>
<p>The live trading ledger, application behavior, and prediction models were not changed by this study. Eight execution checks passed: cost arithmetic, close-confirmed triggers, next-open fills, gap overshoot, stop priority, canceled adds after a rebound, no future-bar leakage into past actions, and partial/trailing exits.</p>
<ul><li><a href="study_config.json">Fixed study configuration and policies</a></li><li><a href="mrcy_summary.json">MRCY reconstruction and arithmetic</a></li><li><a href="event_outcomes.json">Daily event outcome distributions</a></li><li><a href="minute_summary.json">Minute policy results and cost sensitivity</a></li><li><a href="minute_paired_comparisons.json">Paired uncertainty diagnostics</a></li><li><a href="minute_coverage.json">Minute coverage and exclusions</a></li><li><a href="five_year_input_manifest.json">Daily source hashes</a></li><li><a href="simulation.py">Simulation engine</a> · <a href="test_simulation.py">Execution checks</a></li></ul>
<p class="note">Run analysis with the project Python environment: prepare_five_year.py → analyze_daily.py and analyze_minute.py → analyze_mrcy.py → render_report.py. Source downloads are cached under source_data/. The plotting package was installed in a temporary directory without changing project dependencies. Charts are embedded in this HTML and also saved as PNG/SVG for export. Linked result files remain separate local artifacts.</p>
<footer><strong>Primary references</strong><ul>{''.join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t,u in sources)}</ul><p class="note">Prepared November 6, 2026. Local research artifact; no orders placed or automated trading rules enabled.</p></footer>
</main></body></html>'''
for chart in ('mrcy_replay', 'recovery_and_risk', 'risk_return_comparison', 'uncertainty', 'ladder_risk_math'):
    encoded = base64.b64encode((ROOT / f'{chart}.png').read_bytes()).decode('ascii')
    report = report.replace(f'src="{chart}.png"', f'src="data:image/png;base64,{encoded}"')
(ROOT/'report.html').write_text(report)
print('Rendered report and five PNG/SVG figures:',ROOT/'report.html')
