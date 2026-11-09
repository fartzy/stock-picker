"""Analyze all saved Rank scores against independently fetched session prices.

Descriptive research only. Uses original scores, all news states, and all saved
top-20 names regardless of actual execution. Requires a complete price snapshot.
"""
from collections import defaultdict
from datetime import date
import hashlib
import html
import json
from pathlib import Path
from statistics import correlation as reference_correlation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ERA_START = '2026-09-28'
BOOTSTRAPS = 3000
SCORE_BINS = [('<0.20', -np.inf, .2), ('0.20–<0.30', .2, .3),
              ('0.30–<0.40', .3, .4), ('0.40–<0.60', .4, .6), ('≥0.60', .6, np.inf)]


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    x, y = x-x.mean(), y-y.mean()
    denominator = np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/denominator) if denominator > 1e-14 else None


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    out = np.empty(len(values))
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        out[order[start:end]] = (start+1+end)/2
        start = end
    return out


def resamples(days):
    weeks = defaultdict(list)
    for index, day in enumerate(days):
        weeks[date.fromisoformat(day).isocalendar()[:2]].append(index)
    blocks = list(weeks.values())
    rng = np.random.default_rng(20261109)
    for _ in range(BOOTSTRAPS):
        yield np.concatenate([blocks[index] for index in rng.integers(0,len(blocks),len(blocks))])


def interval(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return list(map(float, np.quantile(finite, [.025,.975]))) if finite else None


def stats(rows, bootstrap=False):
    if not rows:
        return {'n':0}
    x = np.array([r['score'] for r in rows])
    y = np.array([r['return'] for r in rows])
    result = {'n':len(rows), 'days':len({r['date'] for r in rows}),
              'pearson':correlation(x,y), 'spearman':correlation(ranks(x),ranks(y)),
              'mean_return_pct':float(y.mean()*100), 'median_return_pct':float(np.median(y)*100),
              'win_pct':float(np.mean(y>0)*100)}
    if bootstrap:
        result['pearson_week_bootstrap_95'] = interval([
            correlation(x[indexes],y[indexes]) for indexes in resamples([r['date'] for r in rows])])
    return result


def residuals(matrix):
    # Alternating projections fit additive date/rank effects even with a missing
    # cell. For a balanced panel this is ordinary two-way demeaning.
    result=matrix.copy()
    for _ in range(100):
        before=result.copy()
        result-=np.nanmean(result,axis=1,keepdims=True)
        result-=np.nanmean(result,axis=0,keepdims=True)
        if np.nanmax(np.abs(result-before))<1e-12:
            return result
    raise ValueError('Date/rank residualization did not converge')


def panel_stats(rows):
    days = sorted({r['date'] for r in rows})
    panel = [sorted([r for r in rows if r['date']==day],key=lambda r:r['rank']) for day in days]
    scores=np.full((len(days),20),np.nan)
    returns=np.full((len(days),20),np.nan)
    for i,day in enumerate(panel):
        for row in day:
            scores[i,row['rank']-1]=row['score']
            returns[i,row['rank']-1]=row['return']
    pairs=[(x[np.isfinite(x)],y[np.isfinite(x)]) for x,y in zip(scores,returns)]
    daily_p=[correlation(x,y) for x,y in pairs]
    daily_s=[correlation(ranks(x),ranks(y)) for x,y in pairs]
    valid=np.isfinite(scores)
    partial=correlation(residuals(scores)[valid],residuals(returns)[valid])
    # Independent least-squares check of the missing-cell fixed-effects fit.
    day_index,rank_index=np.where(valid)
    design=np.column_stack([np.eye(len(days))[day_index],np.eye(20)[rank_index]])
    for matrix in (scores,returns):
        y=matrix[valid]
        fitted=design@np.linalg.lstsq(design,y,rcond=None)[0]
        assert np.allclose(y-fitted,residuals(matrix)[valid],atol=1e-10)
    partial_samples=[]
    for indexes in resamples(days):
        mask=np.isfinite(scores[indexes])
        partial_samples.append(correlation(residuals(scores[indexes])[mask],residuals(returns[indexes])[mask]))
    return {'days':len(days),'observations':len(rows),'mean_within_day_pearson':float(np.mean(daily_p)),
            'mean_within_day_spearman':float(np.mean(daily_s)),
            'after_date_and_rank_pearson':partial,'after_date_and_rank_week_bootstrap_95':interval(partial_samples)}


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def fmt(value, digits=3):
    return '—' if value is None else f'{value:+.{digits}f}'


def main():
    inputs=json.loads((ROOT/'input_rows.json').read_text())
    responses=json.loads((ROOT/'price_snapshot.json').read_text())
    lookup={(r['date'],r['ticker']):r for r in responses}
    old=json.loads((ROOT.parent/'averaging_down_2026_11_06/rank_reconciliation/rank_score_study.json').read_text())
    old_rows={(r['date'],r['ticker']):r for r in old['top20_rows']}
    rows=[]
    unavailable=[]
    for row in inputs:
        if not row['completed_session']:
            continue
        result=lookup[(row['date'],row['ticker'])]
        if 'payload' not in result:
            unavailable.append({**row,'reason':'No verified daily OHLC outcome from provider','provider_error_type':result.get('error_type')})
            continue
        p=result['payload']
        assert p['from']==row['date'] and p['symbol']==row['ticker'].replace('-','.') and p['status']=='OK'
        prior=old_rows[(row['date'],row['ticker'])]
        assert prior['score']==row['score']
        rows.append({**row,'api_open':p['open'],'api_close':p['close'],'api_high':p['high'],'api_low':p['low'],
                     'return':p['close']/p['open']-1,'saved_open_error_pct':100*(row['saved_open']/p['open']-1),
                     'prior_study_return':prior['close_return'],
                     'return_difference_vs_prior_study_pp':100*(p['close']/p['open']-1-prior['close_return'])})
    expected=sum(r['completed_session'] for r in inputs)
    assert len(rows)+len(unavailable)==expected==720
    assert len({(r['date'],r['ticker']) for r in rows})==len(rows)
    eras={}
    for label, start in [('all_saved_history','0000'),('current_objective_era',ERA_START)]:
        selected=[r for r in rows if r['date']>=start]
        untraded=[r for r in selected if not r['traded_same_day']]
        traded=[r for r in selected if r['traded_same_day']]
        eras[label]={'start':min(r['date'] for r in selected),'end':max(r['date'] for r in selected),
                     'all':stats(selected,True),'untraded':stats(untraded,True),'traded':stats(traded,True),
                     'panel':panel_stats(selected),'rank1':stats([r for r in selected if r['rank']==1],True),
                     'without_saved_open_mismatch_over_0_5pct':stats([r for r in selected if abs(r['saved_open_error_pct'])<=.5],True),
                     'score_buckets':[{'score_band':name,'all':stats([r for r in selected if low<=r['score']<high]),
                                       'untraded':stats([r for r in untraded if low<=r['score']<high])}
                                      for name,low,high in SCORE_BINS]}
    for label, era in eras.items():
        selected=[r for r in rows if r['date']>=era['start']]
        independent=reference_correlation([r['score'] for r in selected],[r['return'] for r in selected])
        assert abs(independent-era['all']['pearson'])<1e-12
    price_changes=[r for r in rows if abs(r['return_difference_vs_prior_study_pp'])>.01]
    mismatches=[r for r in rows if abs(r['saved_open_error_pct'])>.5]
    summary={'generated_at':__import__('datetime').datetime.now().astimezone().isoformat(),
             'coverage':{'requested_rows':expected,'complete_rows':len(rows),'complete_sessions':len({r['date'] for r in rows}),
                         'unique_tickers':len({r['ticker'] for r in rows}),
                         'untraded_rows':sum(not r['traded_same_day'] for r in rows),
                         'traded_rows':sum(r['traded_same_day'] for r in rows),
                         'today_pending_rows':sum(not r['completed_session'] for r in inputs),
                         'verified_provider_responses':len(rows),'missing_prices':len(unavailable)},
             'eras':eras,'price_checks':{'rows_different_from_prior_study_by_over_0_01pp':len(price_changes),
                                       'saved_open_mismatch_over_0_5pct':len(mismatches)},
             'methods':[
                 'Every saved completed-session Rank pick is included, regardless of trades or news flags; one row per ticker/date.',
                 'Return = provider unadjusted regular-session close/open - 1. No costs or dividends; no personal trade timing or next-open rule.',
                 'Original recorded scores are retained. No historical rescoring, retraining or synthetic extra scores.',
                 'Traded means any logged execution for that ticker on the same New York date. This is a descriptive stratum, not a causal comparison.',
                 'September 28 separates the documented ListFold objective era. Exact Rank model versions were not retained reliably; nightly models still vary.',
                 'Bootstrap intervals resample whole weeks 3000 times; only six current-era weeks and eight overall weeks make uncertainty estimates fragile.',
                 'Alternating date/rank demeaning handles the missing outcome and is independently checked against least squares. This is an exploratory linear diagnostic, not held-out predictive validation.',
                 'Only the selected top 20 scores were saved. Full-universe correlation cannot be recovered from price history alone.',
                 'Scores can have been produced from stale opening inputs. Provider prices correct evaluation outcomes, not the features used to generate those scores.',
                 'Historical open entries can precede publication. These are stock-performance measurements, not executable strategy backtests.',
                 'Current day excluded because its closing outcome is not known. WSO.B on October 13 has no provider daily record and remains explicitly unavailable; no prior-day price or zero return is substituted. No record is removed because it lost money.'
             ],
             'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['input_rows.json','rank_scan_snapshot.json','price_snapshot.json']}}
    (ROOT/'analysis_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (ROOT/'unavailable_rows.json').write_text(json.dumps(unavailable,indent=2)+'\n')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'price_differences.json').write_text(json.dumps({'prior_study':price_changes,'saved_open':mismatches},indent=2)+'\n')
    render(summary,rows)
    print(json.dumps({'coverage':summary['coverage'],'price_checks':summary['price_checks'],
                      'eras':{k:{j:v[j] for j in ['all','untraded','traded','panel','rank1','without_saved_open_mismatch_over_0_5pct']} for k,v in eras.items()}},indent=2),flush=True)


def render(summary,rows):
    recent=summary['eras']['current_objective_era']
    selected=[r for r in rows if r['date']>=ERA_START]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.4),gridspec_kw={'width_ratios':[1.3,1]})
    for traded,color,label in [(False,'#7084a2','No logged trade that day'),(True,'#684bc1','Logged trade that day')]:
        subset=[r for r in selected if r['traded_same_day']==traded]
        axes[0].scatter([r['score'] for r in subset],[100*r['return'] for r in subset],s=21,alpha=.63,c=color,label=f'{label} (n={len(subset)})',edgecolors='none')
    axes[0].axhline(0,color='#a8b2c3',linewidth=.8)
    axes[0].set(xlabel='Original saved Rank score',ylabel='Verified open-to-close return (%)',title=f"All {len(selected)} later-era picks · Pearson r = {recent['all']['pearson']:+.3f}")
    axes[0].legend(fontsize=8,loc='best')
    groups=recent['score_buckets']; x=np.arange(len(groups))
    for offset,key,color in [(-.18,'all','#684bc1'),(.18,'untraded','#9baec5')]:
        vals=[g[key]['mean_return_pct'] if g[key]['n'] else 0 for g in groups]
        axes[1].bar(x+offset,vals,.34,label='All picks' if key=='all' else 'Untraded picks',color=color)
    axes[1].axhline(0,color='#a8b2c3',linewidth=.8)
    axes[1].set_xticks(x,[g['score_band'] for g in groups],rotation=20)
    axes[1].set(ylabel='Mean open-to-close return (%)',xlabel='Raw-score band',title='Score magnitude · descriptive averages')
    axes[1].legend(fontsize=8)
    fig.suptitle('Every saved pick counts — including stocks never traded',x=.06,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.02,'September 28–November 6, 2026 · 599 verified outcomes · fresh Massive API prices · all news states\nSelected top-20 sample only; zero costs; one provider outcome unavailable. Correlation does not establish a usable forecast.',fontsize=8.5,color='#536078')
    fig.tight_layout(rect=[.02,.09,.99,.91])
    fig.savefig(ROOT/'score_outcomes.png',dpi=170)
    fig.savefig(ROOT/'score_outcomes.svg')
    plt.close(fig)
    cohort_table=table(['Period / cohort','Picks','Days','Pearson r','95% week-bootstrap interval','Mean return'],[
        [f"{era['start']}–{era['end']} / {name}",era[key]['n'],era[key]['days'],fmt(era[key]['pearson']),
         ' to '.join(fmt(v) for v in era[key]['pearson_week_bootstrap_95']),fmt(era[key]['mean_return_pct'],2)+'%']
        for era in summary['eras'].values() for key,name in [('all','all'),('untraded','untraded'),('traded','traded')]])
    bins_table=table(['Score band','All picks','All mean return','Untraded picks','Untraded mean return'],[
        [g['score_band'],g['all']['n'],fmt(g['all'].get('mean_return_pct'),2)+'%',g['untraded']['n'],fmt(g['untraded'].get('mean_return_pct'),2)+'%'] for g in groups])
    p=recent['panel']
    ledger=table(['Date','Ticker','Rank','Score','Logged trade that day','API open','API close','Return','Saved-open difference'],[
        [r['date'],r['ticker'],r['rank'],f"{r['score']:.4f}",'yes' if r['traded_same_day'] else 'no',f"{r['api_open']:.4f}",f"{r['api_close']:.4f}",fmt(r['return']*100,2)+'%',fmt(r['saved_open_error_pct'],2)+'%'] for r in rows])
    svg=(ROOT/'score_outcomes.svg').read_text(); svg=svg[svg.index('<svg'):]
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All saved Rank picks — score and historical returns</title>
<style>body{{background:#f4f6fa;color:#17243c;font:16px/1.6 system-ui;margin:0}}main{{max-width:1160px;margin:auto;padding:44px 28px}}h1{{font-size:38px;line-height:1.15;max-width:900px}}h2{{margin-top:38px}}.lead{{font-size:20px;max-width:960px}}.card{{background:white;padding:22px 28px;border:1px solid #dde3ee;border-radius:14px;margin:24px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #e1e6ef}}th:first-child,td:first-child{{text-align:left}}th{{background:#ecf0f8}}svg{{max-width:100%;height:auto}}a{{color:#6042b2}}.muted{{color:#58677e}}.scroll{{overflow:auto}}summary{{cursor:pointer;font-weight:650}}</style><main>
<p class="muted">November 9, 2026 · Original saved scores · Independently refreshed market prices</p>
<h1>All 720 historical Rank picks checked, including 583 without a logged trade</h1>
<p class="lead">The full available history covers 36 completed sessions and 499 distinct tickers. The Massive API verified 719 outcomes; one WSO.B outcome is unavailable in both daily-summary and daily-aggregate responses. Today's 20 picks remain pending. Inclusion does not depend on whether a stock was traded or whether its news check passed.</p>
<div class="card"><strong>Later objective era: September 28–November 6</strong><p>Across 599 verified picks, raw score versus same-day return has Pearson correlation <strong>{recent['all']['pearson']:+.3f}</strong>. Among the 484 verified picks with no logged trade that day it is <strong>{recent['untraded']['pearson']:+.3f}</strong>. After removing trading-date and ordinal-rank effects, the correlation is <strong>{p['after_date_and_rank_pearson']:+.3f}</strong>, with a week-bootstrap interval of {' to '.join(fmt(v) for v in p['after_date_and_rank_week_bootstrap_95'])}. The numeric score has not established reliable additional predictive value in this sample.</p></div>
<div class="card">{svg}</div><h2>All picks, traded and untraded</h2><div class="scroll">{cohort_table}</div>
<p class="muted">“Traded” means any logged buy or sell for that ticker on the same New York calendar date. These are stock open-to-close returns, not the user's trade P&amp;L. The traded/untraded split is descriptive; trading choices are not random.</p>
<h2>Score bands across all 599 verified later-era picks</h2><div class="scroll">{bins_table}</div><p>These bands pool rank positions and dates. Their sample sizes differ and their returns are exploratory, not validated thresholds or portfolio recommendations.</p>
<h2>Price reconciliation</h2><p>All 719 valid responses match the requested date and ticker and contain valid OHLC prices. WSO.B (October 13, rank 16, untraded) remains unavailable and is excluded from calculations. {summary['price_checks']['rows_different_from_prior_study_by_over_0_01pp']} outcome returns differ from the earlier study by more than 0.01 percentage point. {summary['price_checks']['saved_open_mismatch_over_0_5pct']} original scan opens differ from API opens by more than 0.5%. The evaluation uses the verified API open and close while preserving the original model score.</p>
<h2>Coverage limit</h2><p>The application saved 20 Rank scores per day even when it scored roughly 2,000 stocks. Historical prices cannot recreate the unsaved model scores. Exact Rank model identities are also not reliably retained. The earlier 600-observation study already included untraded picks; this audit extends the presentation to the entire 720-row history and independently refreshes every outcome price.</p>
<h2>Method and limitations</h2><ul>{''.join('<li>'+html.escape(note)+'</li>' for note in summary['methods'])}</ul>
<p>Provider specification: <a href="https://massive.com/docs/rest/stocks/aggregates/daily-ticker-summary">Daily Ticker Summary</a>, with <code>adjusted=false</code>. Original input snapshots, sanitized provider responses, derived rows, and source fingerprints accompany this report.</p>
<details class="card"><summary>Complete 719-outcome ledger (one unavailable outcome listed separately)</summary><div class="scroll">{ledger}</div></details>
<p><a href="summary.json">Summary JSON</a> · <a href="analysis_rows.json">Every analyzed row</a> · <a href="price_snapshot.json">All API responses</a> · <a href="price_differences.json">Price differences</a> · <a href="unavailable_rows.json">Unavailable outcome</a></p></main></html>'''
    (ROOT/'report.html').write_text(report)


if __name__ == '__main__':
    main()
