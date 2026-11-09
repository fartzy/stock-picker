"""Standalone zero-cost report and figure for the recorded Rank-score audit."""
from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/private/tmp/stock-picker-audit-plotting')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/stock-picker-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def usd(value):
    return f'−${abs(value):,.2f}' if value < 0 else f'${value:,.2f}'


def money(value):
    return f'<span class="{"loss" if value < 0 else "gain"}">{usd(value)}</span>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def main():
    study = json.loads((ROOT / 'rank_score_study.json').read_text())
    assert study['assumed_cost_each_way'] == 0
    all_dates = study['eras']['all_dates']
    recent = study['eras']['after_documented_objective_change']
    labels = ['Below 0.4', '0.4 to below 0.6', '0.6 and above']
    def bins(groups):
        return table(['Morning score', 'Positions', 'Profitable', 'Average P&L per $10K', 'Median P&L per $10K'], [
            [label, r['n'], f"{r['wins']} / {r['n']}", money(r['mean_strategy_pnl_10k']), money(r['median_strategy_pnl_10k'])]
            for label, r in zip(labels, groups)
        ])
    def find(feature, outcome, era=recent):
        return next(r for r in era['rank1_correlations'] if r['feature'] == feature and r['outcome'] == outcome)
    def ci(values):
        return f'{values[0]:+.2f} to {values[1]:+.2f}'
    tests = [('Raw score → strategy return', 'score', 'strategy_return'),
             ('Raw score → same-day close return', 'score', 'close_return'),
             ('Gap between #1 and #2 → strategy return', 'gap_to_rank2', 'strategy_return'),
             ('Score standardized within top 20 → strategy return', 'top20_z', 'strategy_return'),
             ('Standardized gap → strategy return', 'gap_to_rank2_z', 'strategy_return'),
             ('Raw score → worst intraday return', 'score', 'intraday_worst_return')]
    correlations = table(['Relationship, later 30 dates', 'Pearson r', '95% week-resampling interval', 'Spearman rho'], [
        [label, f"{r['pearson']:+.3f}", ci(r['pearson_week_bootstrap_95']), f"{r['spearman']:+.3f}"]
        for label, feature, outcome in tests for r in [find(feature, outcome)]
    ])
    add_table = table(['Allow the add when…', 'Added positions, all 36', 'P&L per $10K maximum, all 36', 'Added positions, later 30', 'P&L per $10K maximum, later 30'], [
        [label, a['added_positions'], money(a['pnl_per_10k_maximum']), b['added_positions'], money(b['pnl_per_10k_maximum'])]
        for label, a, b in zip(['Any score', 'Score ≥ 0.4', 'Score ≥ 0.6'], all_dates['add_gate_results'], recent['add_gate_results'])
    ])
    recovery_table = table(['Morning score', 'Later Rank 1 trades reaching −5%', 'Reached combined break-even after a $5K add'], [
        [label, r['n'], f"{r['recovered_partial_add_be']} / {r['n']}"]
        for label, r in zip(labels, recent['drawdown_score_groups'])
    ])
    zero = study['zero_cost_existing_grid']
    zero_table = table(['Strategy', 'Maximum reserved cash', 'P&L at that size', 'P&L per $10K maximum'], [
        ['Original scheduled exit', '$10,000.00', money(2708.87), money(2708.87)],
        *[[label, usd(row['budget']), money(row['pnl']), money(row['pnl_on_10k_reserved_budget'])] for label, row in
          [('Previously selected partial-add candidate', zero['preselected_partial_add']),
           ('Best no-add variant after inspecting all dates', zero['best_no_add_after_inspecting_full_sample']),
           ('Best zero-cost grid setting after inspecting all dates', zero['best_after_inspecting_full_sample'])]],
    ])
    ledger = table(['Date', 'Ticker', 'Morning score', 'Strategy P&L / $10K', 'Same-day return', 'Largest intraday decline'], [
        [r['date'], r['ticker'], f"{r['score']:.4f}", money(r['strategy_pnl_10k']), f"{100*r['close_return']:+.2f}%", f"{100*r['intraday_worst_return']:+.2f}%"]
        for r in study['rank1_rows']
    ])
    selected = [r for r in study['rank1_rows'] if r['date'] >= '2026-09-28']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.parse_math': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.4), gridspec_kw={'width_ratios': [1.25, 1]})
    colors = ['#6852b6' if r['score'] < .6 else '#b36f53' for r in selected]
    axes[0].scatter([r['score'] for r in selected], [100*r['strategy_return'] for r in selected], c=colors, s=43, alpha=.85, edgecolor='white', linewidth=.6)
    axes[0].axhline(0, color='#687586', linewidth=.8)
    axes[0].axvline(.6, color='#b36f53', linestyle='--', linewidth=1)
    axes[0].set_title('Raw score versus the existing exit strategy', loc='left', fontsize=12)
    axes[0].set_xlabel('Recorded morning Rank 1 score')
    axes[0].set_ylabel('Position return (%)')
    axes[0].set_ylim(-18, 19)
    axes[0].set_xlim(.16, 1.3)
    axes[0].set_xticks([.2, .4, .6, .8, 1, 1.2])
    axes[0].grid(axis='y', alpha=.14)
    axes[0].text(.19, 16.3, 'Pearson r = +0.014 · 30 mornings', color='#536078', fontsize=10)
    axes[0].text(.61, -16.7, '0.6 threshold', color='#a05d45', fontsize=9)
    for ticker, dx, dy in [('CTVA', .025, -.4), ('IMO', .025, .4), ('TSM', .025, -.5)]:
        r = next(x for x in selected if x['ticker'] == ticker)
        axes[0].text(r['score']+dx, r['strategy_return']*100+dy, ticker, fontsize=9, color='#536078')
    means = [r['mean_strategy_pnl_10k'] for r in recent['rank1_score_groups']]
    axes[1].bar([0, 1, 2], means, color=['#9484ce', '#6852b6', '#b36f53'], width=.6)
    axes[1].set_xticks([0, 1, 2], ['Below 0.4', '0.4 to <0.6', '0.6 or higher'])
    axes[1].set_ylabel('Average P&L per $10K position ($)')
    axes[1].set_title('The higher-score group did not earn more', loc='left', fontsize=12)
    axes[1].set_ylim(0, 385)
    axes[1].grid(axis='y', alpha=.14)
    for i, r in enumerate(recent['rank1_score_groups']):
        axes[1].text(i, means[i]+10, f"{usd(means[i])}\nn = {r['n']}", ha='center', fontsize=10)
    fig.suptitle('Does a higher Rank score predict a better outcome?', x=.065, ha='left', fontsize=17, fontweight='bold')
    fig.text(.065, .025, 'September 28–November 6, 2026 · 0% assumed costs · recorded scores and corrected prices\n'
             'Close if flat/up from entry; otherwise next open. Small exploratory sample; score groups are not calibrated probabilities.', color='#536078', fontsize=9)
    fig.tight_layout(rect=[.01, .12, .99, .91], w_pad=3)
    fig.savefig(ROOT / 'rank_score_study.png', dpi=180, bbox_inches='tight')
    fig.savefig(ROOT / 'rank_score_study.svg', bbox_inches='tight')
    plt.close(fig)
    svg = (ROOT / 'rank_score_study.svg').read_text(); svg = svg[svg.index('<svg'):]
    notes = ''.join(f'<li>{escape(note)}</li>' for note in study['method'])
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rank score and outcomes · zero assumed costs</title>
<style>
body{{margin:0;background:#f4f6fa;color:#203047;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:38px 24px 65px}}
h1{{font-size:36px;line-height:1.2;max-width:1000px}}h2{{font-size:23px;margin-top:36px}}p{{max-width:1060px}}.lead{{font-size:20px}}
.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#6551a5;font-weight:700}}.callout{{background:#eae6f8;border-left:4px solid #7b68bd;padding:16px 22px;border-radius:5px}}
.note{{color:#536078}}.scroll{{overflow:auto;background:white;border:1px solid #dce1eb;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}
th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid #e7ebf2}}th{{background:#eef1f7}}th:first-child,td:first-child{{text-align:left}}
.gain{{color:#187a61;font-weight:600}}.loss{{color:#a4435b;font-weight:600}}a{{color:#6250a4}}li{{margin-bottom:8px}}svg{{width:100%;height:auto;background:white;border-radius:12px}}
details{{margin-top:24px}}summary{{cursor:pointer;font-weight:650}}footer{{border-top:1px solid #dce1eb;margin-top:36px;padding-top:18px;font-size:13px;color:#536078}}code{{font-size:13px}}
</style></head><body><main>
<div class="eyebrow">Recorded Rank scores · November 7, 2026 research · 0% assumed costs</div>
<h1>Raw score has not established a reliable confidence threshold</h1>
<p class="lead">Yes, the numerical score can be tested. In the 30 Rank 1 mornings after the documented ranking-objective change, raw score and the close/next-open strategy return have Pearson correlation +0.014 and Spearman correlation +0.061. This sample does not support treating 0.6 as a stronger reason to enter or add than 0.4.</p>
<div class="callout"><strong>No execution-cost deduction:</strong> all results in this report use exactly 0% assumed spread, slippage and fees. Historical aggregate-price fills remain hypothetical; removing an assumed deduction does not verify their execution.</div>
<h2 id="meaning">What the score actually means</h2>
<p>The Rank model learns to order stocks within a day. Its raw score is neither a probability nor a predicted percentage return. The shared storage field is named <code>predicted_return</code>, but the Rank serving path sends the raw ranking output through it.</p>
<p>The implemented ListFold loss is unchanged if the same constant is added to every stock’s score. A list scored 0.6, 0.5, 0.4 has the same ordering and pairwise differences after a shift to 0.8, 0.7, 0.6. Therefore absolute thresholds require separate empirical calibration. This does not prove score levels contain no useful information; it explains why that usefulness cannot be assumed.</p>
<p>See <code>python/stock_picker/training/model.py:227</code>, <code>listwise.py:91</code>, <code>buy_signal.py:234</code>, and the <a href="https://arxiv.org/abs/2104.12484">ListFold paper</a>. The objective changed from LambdaRank to ListFold on September 26, per <code>docs/adr/0018-listfold-rank-objective.md:1</code>. Later models were also retrained and changed features, so even the later dates are not one fixed artifact.</p>
<h2 id="groups">The direct 0.4 versus 0.6 comparison</h2>
<p>Rank 1 only, September 28–November 6: same $10K starting position, same close/next-open exit, no cost deduction. Every result belongs to its original morning score; no historical prices were rescored using today’s model.</p>
{bins(recent['rank1_score_groups'])}
<p>The 0.6-and-above group contains only six positions. Its lower average does not establish that low scores are inherently better, and it is not a reason to invert the signal. It does mean this sample does not validate “a higher score deserves more money.”</p>
<h2 id="correlations">Score, score gaps, and standardized scores</h2>
{correlations}
<p>Pearson tests linear association; Spearman tests whether larger values tend to accompany larger outcomes. Same-day close returns—the model’s training target—show a modest positive monotone association (rho +0.274), with a week-resampling interval of −0.009 to +0.598. That interval includes zero. The association is weaker for the actual close/next-open exit strategy.</p>
<p class="note">The intervals resample whole trading weeks 3,000 times. Only six later weeks exist, so these are sensitivity estimates, not precise guarantees. Several exploratory relationships were examined; no multiple-testing-adjusted discovery is claimed. Standardized scores use only the saved top 20, not the full universe.</p>
<div role="img" aria-label="Scatter of morning Rank score versus return, and the average return of each score group, using zero costs.">{svg}</div>
<h2 id="ordinal">Does the magnitude add information beyond ordinal rank?</h2>
<p>Across the later 600 top-20 observations, the average within-day rank correlation with same-day returns is +0.087. This is a small positive ordering association, not proof that a score of 0.6 has a calibrated meaning across days. Within-day Spearman uses ordering, so it cannot by itself answer whether the numerical magnitude adds information beyond #1, #2, and so on.</p>
<p>Removing date and ordinal-rank effects from that balanced panel leaves a raw-score/return correlation of −0.039; its week-resampling interval is −0.140 to +0.119. This simple linear diagnostic finds no clear additional positive association. It does not rule out a nonlinear relationship or better calibration on a larger sample.</p>
<h2 id="drawdown">Does a high morning score make averaging down safer?</h2>
<p>Only seven later-period Rank 1 positions reached a 5% decline. The following checks ask whether one $5K addition to a $10K initial position could reach combined break-even before the original scheduled exit, using the two documented OHLC path sensitivities.</p>
{recovery_table}
<p>The two high-score cases both recovered, but two observations cannot establish a reliable recovery probability. Raw score versus return from the first add price to the scheduled exit has correlation −0.103 across the seven cases, with a very wide interval. These use the original morning score; there is no recorded model update after the decline.</p>
<h2 id="gate">Testing a score gate on the partial-add rule</h2>
<p>Keep the previously selected rule: $10K initially, add $5K once after a 5% drop, $2K total-loss trigger, and the scheduled close/next-open exit. The only change is whether the morning score permits that addition. Every version retains the loss trigger, trades every original pick, reserves up to $15K, and uses zero costs.</p>
{add_table}
<p>Requiring a score of at least 0.6 reduced profit in both date ranges. These are historical comparisons with small triggered samples; they do not validate a future score filter or prove the opposite filter is optimal.</p>
<h2 id="rank2">Rank 2 cross-check</h2>
<p>The later Rank 2 sample has 29 completed positions; November 6 SLF is awaiting the next open and is excluded from return calculations. Its raw-score/strategy-return correlation is −0.172, also with a wide interval spanning zero.</p>
{bins(recent['rank2_score_groups'])}
<h2 id="all-dates">Why the earlier algorithm matters</h2>
<p>Pooling all 36 Rank 1 mornings gives correlation −0.192, versus +0.014 after the documented change. The six earlier mornings all have scores above 0.8, and include large losses. That makes a pooled “high score” comparison partly a comparison of model eras.</p>
{bins(all_dates['rank1_score_groups'])}
<p>Exact Rank-artifact identity is not reliably stored in these scan snapshots. The payload’s <code>model_run_id</code> can refer to the separately selected Fit run, while the Rank path independently loads <code>day_session_return_rank</code>. Consequently the later slice is an era check, not exact per-version calibration.</p>
<h2 id="zero-cost">The earlier averaging-down comparison, with zero costs</h2>
{zero_table}
<p>The previously chosen partial-add rule makes $4,645.45 on a $15K maximum budget, or $3,096.97 when scaled to a $10K maximum. The best no-add exit variant in the existing grid makes $3,287.16. Searching the full zero-cost grid favors an 8% trigger with a $2,500 add, $1,500 total-loss trigger and $2,000 recovery-profit exit, at $3,324.67 per $10K reserved. Those full-sample winners remain retrospective choices; changing the assumed costs does not supply new independent evidence.</p>
<h2 id="next">What would make the score useful as a confidence measure?</h2>
<p>Record the exact Rank artifact and the distribution of all morning scores. Test raw score, score gaps, and relative position within that distribution against later realized outcomes, with training and evaluation separated chronologically. Recovery after a decline should have its own conditional target. A calibrated probability or expected return would then need its own prospective validation; the raw Rank number should not simply be relabeled as confidence.</p>
<details><summary>Complete Rank 1 ledger</summary>{ledger}</details>
<details><summary>Method, limitations, and verification</summary><ul>{notes}</ul>
<p>All 720 saved scores and completed-session returns reconcile. The 36 Rank 1 audit records sum to $2,708.87, and the preselected zero-cost partial-add replay matches the earlier grid. Three price recomputations differ from the frozen dashboard-audit values by one cent; both values are retained. A missing verified next-open price for GRAL is not needed because its entry-day close is positive. No missing required exit is substituted.</p>
<p>The correlation definitions follow <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html">Pearson documentation</a> and <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html">Spearman documentation</a>; implementation uses the explicit centered-product and average-tie-rank formulas. Hypothetical open/close timing, aggregate-bar order and liquidity limitations remain as in the earlier studies.</p></details>
<footer><a href="rank_score_study.json">Complete results and source hashes</a> · <a href="rank_score_study.png">Export figure</a> · Reproduce with <code>rank_score_study.py</code> then <code>render_rank_score_study.py</code>.</footer>
</main></body></html>
'''
    (ROOT / 'rank_score_report.html').write_text(html)
    print('Wrote rank_score_report.html and rank_score_study.png/.svg')


if __name__ == '__main__':
    main()
