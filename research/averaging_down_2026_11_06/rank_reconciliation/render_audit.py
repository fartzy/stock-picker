"""Render the frozen calculation audit and its matched-policy comparison."""
from pathlib import Path
import html
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/private/tmp/stock-picker-audit-plotting')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/stock-picker-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name):
    return json.loads((ROOT / name).read_text())


def usd(n):
    return f'−${abs(n):,.2f}' if n < 0 else f'${n:,.2f}'


original = load('dashboard_summary.json')
verified = load('verified_price_summary.json')
repair = load('paper_price_repair_summary.json')
rank1 = load('rank1_significance.json')
matched = load('verified_matched_summary.json')
old = json.loads((ROOT.parent / 'conditional_overnight_summary.json').read_text())
old = next(r for r in old if r['split'] == 'holdout' and r['cost_bps'] == 11 and r['policy'] == 'ladder5_target2_gate3pct')
policies = {r['policy']: r for r in matched['policies'] if r['cost_bps'] == 11}
names = ['dashboard_2500', 'ladder_2500_target2_gate3pct', 'single_2500_target2_gate3pct',
         'dashboard_10000', 'upfront_10000_target2_gate3pct']
labels = {'dashboard_2500': 'Existing close / next-open exits · $2,500',
          'ladder_2500_target2_gate3pct': 'Proposed $2,500 ladder · up to $10,000',
          'single_2500_target2_gate3pct': '2% target / overnight gate · $2,500, no adds',
          'dashboard_10000': 'Existing close / next-open exits · $10,000',
          'upfront_10000_target2_gate3pct': '2% target / overnight gate · $10,000 upfront'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
panels = [
    (axes[0], [84.04185714285715, verified['verified_gross']['mean_per_position'], verified['verified_net11']['mean_per_position']],
     ['Saved dashboard\nprices', 'Verified daily\nprices', 'Verified prices\n+ modeled costs'],
     '70 positions · $10,000 each'),
    (axes[1], [policies['dashboard_2500']['mean_per_position'], policies['ladder_2500_target2_gate3pct']['mean_per_position']],
     ['Existing\nexit rule', 'Proposed\naveraging-down rule'],
     '47 matched positions · $2,500 initially'),
]
for ax, values, ticks, title in panels:
    bars = ax.bar(range(len(values)), values, color=['#6e61ce' if x >= 0 else '#be526e' for x in values], width=.62)
    ax.axhline(0, color='#69758d', linewidth=.8)
    ax.set_xticks(range(len(values)), ticks)
    ax.set_title(title, loc='left', pad=19, fontweight='bold')
    ax.set_ylabel('Average profit per position ($)')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(min(0, min(values)) - max(values) * .12, max(values) * 1.24)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(values) * .045, usd(value),
                ha='center', fontweight='bold', fontsize=11)
fig.suptitle('Separate the price audit from the strategy comparison', x=.06, ha='left', fontsize=16, fontweight='bold')
fig.text(.06, .015, 'Modeled costs: 0.11% per side. The panels use different populations and position sizes; compare bars within each panel.', fontsize=9, color='#536078')
fig.tight_layout(rect=[.02, .07, 1, .9], w_pad=3)
fig.savefig(ROOT / 'comparison.png', dpi=180, bbox_inches='tight')
fig.savefig(ROOT / 'comparison.svg', bbox_inches='tight')
plt.close(fig)
svg = (ROOT / 'comparison.svg').read_text()
svg = svg[svg.index('<svg'):]

def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


price_table = table(['Same 70 saved selections', 'Total P&amp;L', 'Average / position', 'Gain / $20K'], [
    ['Dashboard as displayed', '$5,882.93', '$84.04', '29.41%'],
    ['Verified prices, production rounding', '$2,454.04', '$35.06', '12.27%'],
    ['Verified prices + assumed 0.11% per side', usd(verified['verified_net11']['total_pnl']), usd(verified['verified_net11']['mean_per_position']), '4.55%'],
    ['Verified prices + assumed 0.26% per side', usd(verified['verified_net26']['total_pnl']), usd(verified['verified_net26']['mean_per_position']), '−5.95%'],
])
mismatches = table(['Date', 'Stock', 'Saved entry', 'Verified open', 'Saved P&amp;L', 'Verified-price P&amp;L'], [
    [r['date'], r['ticker'], usd(r['saved_open']), usd(r['verified_open']), usd(r['saved_pnl']), usd(r['pnl'])]
    for r in reversed(verified['significant_open_mismatches'])])
policy_table = table(['Rule / starting size', 'Mean / position', 'Win rate', 'Worst position', 'Avg. peak capital'], [
    [labels[name], usd(policies[name]['mean_per_position']), f"{policies[name]['win_pct']:.1f}%",
     usd(policies[name]['worst']), usd(policies[name]['average_peak_capital_per_position'])] for name in names])
all_rows = table(['Date', 'Rank', 'Stock', 'Saved open', 'Verified open', 'Saved P&amp;L', 'Verified-price P&amp;L'], [
    [r['date'], r['rank'], r['ticker'], f"${r['saved_open']:,.4f}", f"${r['verified_open']:,.4f}", usd(r['saved_pnl']), usd(r['pnl'])]
    for r in load('verified_price_rows.json')])
source_map = table(['Calculation / finding', 'Source location'], [
    ['Position size and gross return', 'typescript/src/whatIf.ts:326–342'],
    ['Period P&amp;L / average daily capital', 'typescript/src/whatIf.ts:289–301; 442–451'],
    ['Down-day selection and next-open increment', 'typescript/src/whatIf.ts:36–74; 101–137'],
    ['Complete-day headline aggregation', 'typescript/src/whatIf.ts:157–184'],
    ['Per-position footer aggregation', 'typescript/src/components/WhatIf.tsx:183–197; 866–888'],
    ['Daily row shows overnight increment', 'typescript/src/components/WhatIf.tsx:509–524'],
    ['Saved scan open takes precedence over daily open', 'python/stock_picker/paper/book.py:169–197'],
    ['Existing test requires the stale scan price', 'python/stock_picker/paper/tests/test_book.py:254–284'],
    ['Real-price regression cases and controls', 'research/averaging_down_2026_11_06/rank_reconciliation/regressions_price_integrity.py:42–78'],
    ['Scoring happens after the open', 'python/stock_picker/training/morning.py:1–4; 285–299'],
    ['Original non-Rank entry population', 'research/averaging_down_2026_11_06/prepare_five_year.py:34–42'],
    ['Original $12.81 result', 'research/averaging_down_2026_11_06/conditional_overnight_summary.json:338–364'],
])

document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rank Top 2 — calculation and strategy audit</title>
<style>
:root{{color-scheme:dark;--bg:#0b1020;--panel:#151d36;--ink:#e9edf8;--muted:#abb8d1;--line:#303c59;--accent:#b5a7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:0 auto;padding:50px 30px 70px}}h1{{font-size:clamp(29px,4vw,43px);line-height:1.12;margin:10px 0 22px;letter-spacing:-1px}}h2{{font-size:23px;margin:42px 0 14px}}h3{{font-size:17px;margin:25px 0 8px}}p{{max-width:92ch}}.eyebrow{{font-size:12px;letter-spacing:1.6px;color:var(--accent);text-transform:uppercase}}.lead{{font-size:20px;color:#d4def3}}.note{{font-size:14px;color:var(--muted)}}.callout{{padding:20px 23px;background:var(--panel);border-left:4px solid var(--accent);border-radius:4px}}.equation{{font-size:19px;font-variant-numeric:tabular-nums;line-height:2}}.scroll{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:right;padding:13px 15px;vertical-align:top;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}th{{background:#1c2642;color:#c7d2eb;font-size:12px}}th:first-child,td:first-child{{text-align:left}}tr:last-child td{{border-bottom:0}}a{{color:var(--accent)}}.figure{{background:white;border-radius:10px;margin:30px 0;overflow:hidden}}svg{{width:100%;height:auto;display:block}}li{{margin:10px 0}}details{{margin:25px 0;padding:18px;background:var(--panel);border-radius:10px}}summary{{cursor:pointer;font-weight:650}}footer{{margin-top:45px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted)}}code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:650px){{main{{padding:26px 16px}}td,th{{padding:10px}}}}@media print{{:root{{color-scheme:light;--bg:white;--ink:#14213b;--muted:#445470;--panel:#f3f5fa;--line:#d2d9e7;--accent:#6555b4}}main{{padding:0}}th{{background:#edf0f7;color:#14213b}}.lead{{color:#243957}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}
</style></head><body><main>
<div class="eyebrow">Stock Picker · Research audit · November 6, 2026</div>
<h1>What the Rank Top 2 result actually says</h1>
<p class="lead">The screenshot represents <strong>$84.04 per completed position</strong> on its saved prices. The earlier $12.81 simulation used different entries, sizes and exits. Comparing those figures as if they measured the same strategy was wrong.</p>
<div class="callout"><strong>The deeper issue is the input data.</strong> Replacing saved opens and closes with verified daily prices for the same 70 picks reduces the production calculation from <strong>$5,882.93 / 29.41%</strong> to <strong>$2,454.04 / 12.27%</strong>. Three large discrepancies are demonstrably previous-session opens. This audit keeps those findings separate from assumed trading costs and from tests of averaging down.</div>
<details open><summary>Confirmed code regression — tested against the introducing change</summary>
<p>New tests use the saved PS (September 21), HHH (September 22) and TRLV (September 23) signals with verified same-session daily bars. All three saved entry prices are below that day's entire trading range. Unadjusted minute bars agree with the daily opens and ranges; corporate-action checks found no splits or dividends across the prior, entry and following sessions.</p>
<p>All <strong>six cases pass</strong> in the paper-book code before commit <code>37357d2465</code> (September 26). After that change, the <strong>three stale-entry cases fail</strong> and the three valid-entry controls pass. Current production code has the same failures. The existing seven pricing and What If suites pass all <strong>114 tests</strong>; one existing test explicitly requires preserving a contradicted scan opening price.</p>
<p>The change gave the archived scan price unconditional priority over the daily opening price. This preserves an input for auditing but incorrectly treats it as a valid hypothetical purchase price. Refreshing the book repeats the error. These tests use temporary stores and leave the original signals untouched.</p>
<p>Correcting only the three confirmed stale opens changes profit to <strong>$3,022.76</strong>, removing <strong>$2,860.17</strong>. Correcting all opening prices gives <strong>$2,454.04</strong>. Correcting closing prices alone leaves <strong>$5,882.93</strong> unchanged. The full <strong>$3,428.89</strong> discrepancy therefore comes from opening prices in this frozen comparison, without any cost assumptions.</p>
<p><a href="regression_versions.json">Before/after test results</a> · <a href="regression_current.txt">Current failing assertions</a> · <a href="regressions_price_integrity.py">Reproduction tests</a> · <a href="isolated_price_impact.json">Isolated price impacts</a> · <a href="isolate_price_impact.mjs">Production calculation replay</a> · <a href="regression_minute_evidence.json">Minute-bar evidence</a> · <a href="regression_corporate_actions.json">Corporate-action checks</a></p>
<p class="note"><strong>Repair applied November 6:</strong> completed results now use a same-session open/close pair. The API was restarted and all 6,656 returned price rows matched the planned repair. Original scans and 232 replay rows were preserved; a database backup was saved. Code commit: <code>17556d830</code>. The live daily cache gives Rank 1 $2,708.87 (27.09%) and Rank 2 $2,451.66 (12.26%). Minor differences from the raw-price audit come from cached daily-price precision. Raw quote payloads from the original mornings were not retained, so the exact upstream HTTP response cannot be reconstructed.</p></details>
<h2>Latest check: is Rank Top 1 statistically convincing?</h2>
<p>For the user's Rank 1 setting—$10K per position, winners sold at the close, losers sold at the following open—the repaired dashboard has <strong>36 positions, $2,708.87 profit and 27.09% period gain</strong>, before costs. There are 25 wins and 11 losses. This remains an opening-price simulation on the final saved selections, not a record of executable fills.</p>
<p>The one-sample t-test of zero mean profit gives <strong>t = {rank1['gross']['t_statistic']:.3f}</strong> with 35 degrees of freedom: <strong>one-sided p = {rank1['gross']['p_positive_mean_one_sided']:.3f}</strong> for a positive mean, and two-sided p = {rank1['gross']['p_two_sided']:.3f}. The mean is <strong>{usd(rank1['gross']['mean'])}</strong> per $10K position. Its 95% confidence interval is <strong>{usd(rank1['gross']['mean_95pct_t_interval'][0])} to {usd(rank1['gross']['mean_95pct_t_interval'][1])}</strong>. This does not meet a conventional 5% significance threshold.</p>
<p>200,000 independent bootstrap resamples give a mean-profit interval of about −$139 to +$270; resampling whole calendar weeks gives about −$159 to +$292. Both include zero. Only eight weeks are available, so the clustered estimate is itself imprecise. The test assumes a stable population and independence more than the data can establish; trying multiple Rank sizes and exit rules makes this an exploratory result.</p>
<p>The win-rate-only binomial test against 50% gives p = {rank1['win_frequency_only']['p_one_sided']:.3f}, but that tests a different question. Average wins are about $365 and average losses about $582, so winning more often than half the time does not establish positive expected profit. The three largest winners total $3,186.60; the other 33 sum to −$477.73. That is a tail-sensitivity diagnostic, not a reason to discard the winning trades.</p>
<p>With the explicitly assumed 0.11% cost each way, mean profit is $53.11 and the one-sided zero-mean test gives p = 0.310. No p-value is the probability that the strategy is merely luck or lacks an edge. A useful confirmation would freeze this rule and evaluate fresh, timestamped forward observations at attainable entry prices, including measured execution costs.</p>
<p class="note"><a href="rank1_significance.json">All statistical results</a> · <a href="rank1_significance_rows.json">The 36 position results</a> · <a href="rank1_significance.py">Reproducible calculations</a> · <a href="live_repaired_totals.json">Live API reconciliation</a> · <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html">SciPy test assumptions</a> · <a href="https://www.amstat.org/asa/files/pdfs/P-ValueStatement.pdf">ASA interpretation guidance</a></p>
<h2>1. Reproducing the screenshot</h2>
<p>I froze the local paper-book and next-open API responses, compiled the actual production TypeScript calculation, and replayed Rank Top 2, $10K per position, no news skips, no blacklist filter, holding losing positions to the next opening price. Both replay rows on October 28 are excluded by the production code.</p>
<div class="equation">35 complete days × 2 positions = <strong>70 positions</strong><br>$5,882.93 ÷ 70 = <strong>$84.04 per position</strong><br>$5,882.93 ÷ $20,000 reused daily capital = <strong>29.41% period gain</strong></div>
<p>The 29.41% is a simple cumulative gain on reused capital. It is neither the average return of one position nor a compounded return. The corresponding mean gross return per position is 0.8404%. Summing $700,000 of repeated purchases would measure turnover, not the distinct cash required.</p>
<p>At the displayed prices, selling everything at the close makes $4,327.47. Holding losing positions overnight adds $1,555.46, for $5,882.93 total. The first-ranked pick contributes $5,352.38; the second-ranked pick contributes $530.55. “Top 2” includes both ranks.</p>
<h2>2. Why the earlier $13 was not the same experiment</h2>
<p>The exact prior estimate was <strong>{usd(old['mean'])} per position</strong> over 208 later-period historical events selected after a daily decline of at least 5%. They were not the Rank morning picks. Each position began with $2,500, could add three equal purchases after successive 5% declines, targeted a 2% net gain, used the proposed loss and overnight gates, and had a two-week outer deadline. Average peak capital was {usd(old['avg_max_capital'])}; modeled friction was 0.11% each way.</p>
<p>Position size alone explains a substantial part of the apparent discrepancy: the saved-price Rank result scales from $84.04 at $10K to $21.01 at $2,500. Applying the same 0.11% per-side assumption makes that $15.47. This arithmetic normalization does <em>not</em> show that the two strategies are equivalent, and it still relies on the erroneous saved prices.</p>
<h2>3. Verifying the input prices</h2>
<p>I retrieved unadjusted daily summaries and minute bars for all 72 selected entries. For the 70 headline positions, eight saved opens differ from the daily source by more than 0.5%. PS, HHH and TRLV match the previous session’s open in the local daily-price files. The paper-book builder preserves a scan’s supplied open and substitutes the daily open only when the scan open is absent.</p>
{mismatches}
<p class="note">P&amp;L above applies the same “sell winners at the close, hold losers to the next open” rule after updating the entry price. PS changes from a day winner to a loser, so its verified-price exit is the following open. The remaining smaller price differences and rounding are included in the total below.</p>
{price_table}
<p><strong>The cost rows are scenarios, not measured brokerage charges.</strong> A 0.11% assumption means about $11 on each side of a $10K trade for modeled spread, slippage and fees. Lower friction improves the result; higher friction worsens it. The approximately $13 in the third row is a numerical coincidence with the earlier $12.81 from a different experiment—it does not validate that earlier comparison.</p>
<p>The production replay rounds individual close P&amp;L and overnight increments to cents and gives $2,454.04. Direct full-precision arithmetic rounds to $2,454.05, a one-cent difference. These are same-selection price checks, not a rerun of the historical ranking model using corrected inputs.</p>
<div class="figure">{svg}</div>
<h2>4. Testing the proposed rule on actual Rank picks</h2>
<p>I compared the rules on <strong>47 identical Rank positions</strong> with verified starting prices and ten completed trading sessions through October 26 entries. Six sparse minute histories and one missing-session history were excluded from every policy. Sixteen newer positions lack the full horizon; two November 6 positions belong to the pending dashboard day. None were admitted or excluded based on whether a policy won.</p>
<p>Every policy uses the same entries and modeled 0.11% per-side friction. Intraday decisions observe a completed minute and execute at the next minute’s open. The proposed gate allows an overnight stay only while the marked loss is within the smaller of $600 and 3% of purchased capital. The target is 2%; the loss decision trigger is $1,200. No split or dividend event was found in any of the 47 matched windows.</p>
{policy_table}
<p>The existing exits with $2,500 positions average <strong>$12.74</strong>. The ladder averages <strong>−$0.19</strong> despite deploying more capital. Its 72% win rate comes with larger losing positions: average win about $78, average loss about $205. The matched average difference versus existing exits is −$12.93; a date/ticker-cluster bootstrap diagnostic spans roughly −$51 to +$21. The sample does not establish a reliable advantage for adding.</p>
<p>The capped-target rule also clips large winners that materially support this strategy. On the saved-price full history, the best five positions contribute $5,908.26—slightly more than the entire $5,882.93 profit. That is a sensitivity diagnostic, not a reason to delete winners: it shows why a 2% profit target must be tested rather than assumed to help.</p>
<h2>5. What was actually executable</h2>
<p>The model consumes the day’s opening price, then publishes its selections. Logs show November 4 Rank publication at 09:34:29 ET and November 6 at 09:35:12 ET. An exact opening-price fill cannot generally be assumed after waiting for those selections.</p>
<p>As a separate timing sensitivity, 51 positions have a publication timestamp and a minute bar within two minutes afterward. Re-entering at the first bar after publication and reassessing the down-day rule produces about <strong>$5,407 gross / $4,274 after assumed friction</strong> on that subset, versus $6,128 gross using verified opens for those same positions. This is encouraging evidence for further testing, not a substitute for the full-history result: the subset is September 28–November 5 and excludes one sparse quote. It also uses the final saved selections; later quote-recovery passes can change the list. It is not an immutable, timestamp-perfect replay.</p>
<h2>6. The other display discrepancies</h2>
<ul>
<li><strong>The headline and footer use different populations.</strong> Today’s KOD winner (+$696.97) is in the footer but the whole November 6 day is excluded from the headline while SLF awaits November 9’s open. Thus the footer has 71 priced picks, 48 wins and about $92.67 average, whereas the headline has 70 positions, 47 wins and $84.04 average.</li>
<li><strong>The collapsed daily row can show only the overnight improvement.</strong> November 5’s +$71.18 is the increment from holding TEM, while that day’s full Rank result is +$749.55. The row uses the full result instead when no position is held. Those labels need to make the distinction explicit.</li>
<li><strong>The SPY comparisons include November 6.</strong> The overnight headline currently ends with November 5 entries. Aligning the completed-date population would make the comparison clearer.</li>
<li><strong>Replay duplicates are not inflating the headline.</strong> Both October 28 replay rows are explicitly excluded.</li>
</ul>
<h2>7. What this means for the loss-budget question</h2>
<p>The current saved-price history has a $2,620.54 drawdown between completed-day P&amp;L peaks and troughs, before intraday or financing effects. The verified-price worst individual position loses about $2,086 before modeled costs. Neither history fits an absolute $1,500 loss limit at $10K per position. Two simultaneous positions and longer overlapping holdings also need a combined cash and loss budget; a per-position limit alone is insufficient.</p>
<p>The useful next research step is to preserve the Rank selection signal, repair or quarantine stale entry prices, use archived publication-time entries, and compare exits with an explicit portfolio risk budget. The analysis so far does not justify calling the proposed averaging-down rule an ideal strategy or replacing the existing rule with it. Thirty-five completed days are also too few to treat the measured mean as a stable forecast; the bootstrap diagnostics include negative mean returns.</p>
<p class="note">Stop triggers do not guarantee a maximum realized loss through gaps or unavailable liquidity. <a href="https://www.finra.org/investors/insights/stop-orders-factors-consider-during-volatile-markets">FINRA’s stop-order guidance</a> explains the execution limitation. Price data were fetched using <a href="https://massive.com/docs/rest/stocks/aggregates/custom-bars">Massive’s documented unadjusted aggregate convention</a>.</p>
<details><summary>Inspect all 70 positions</summary><p class="note">Unrounded price-reconciled results; the production total differs by one cent through line-item rounding.</p>{all_rows}</details>
<details><summary>Code locations and reproducible evidence</summary>{source_map}<p><a href="dashboard_summary.json">Original dashboard replay</a> · <a href="verified_production_math.json">Verified-price production replay</a> · <a href="verified_price_summary.json">Price audit</a> · <a href="verified_matched_summary.json">Matched comparison</a> · <a href="publication_times.json">Publication log references</a> · <a href="matched_corporate_actions.json">Corporate-action checks</a></p><p>Reproduction: compile <code>typescript/src/whatIf.ts</code> with the installed TypeScript compiler into <code>compiled/</code>; run <code>audit_dashboard.mjs</code>, <code>verify_prices.py</code>, then <code>compare_rank_policies.py --verified-prices</code>. Frozen API JSON and raw minute archives sit beside these scripts. The simulation’s 10 execution checks pass; additional reconciliation assertions verify unique entries, exact headline reproduction, arithmetic decomposition and equal policy populations.</p></details>
<footer>This is a research artifact using November 6, 2026 snapshots. The original comparison below the repair notice preserves pre-repair evidence. The live paper book and pricing code have since been repaired as described above. Figures are hypothetical, before tax, financing, and any costs beyond the explicitly modeled friction.</footer>
</main></body></html>'''
(ROOT / 'report.html').write_text(document)
print(ROOT / 'report.html')
