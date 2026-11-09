"""Readable research report and exportable figure for the theoretical search."""
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
    analysis = json.loads((ROOT / 'average_down_grid_analysis.json').read_text())
    original = json.loads((ROOT / 'average_down_grid_original.json').read_text())
    stress = json.loads((ROOT / 'average_down_historical_stress.json').read_text())
    verification = json.loads((ROOT / 'average_down_grid_verification.json').read_text())
    assert verification['status'] == 'passed'
    lookup = {(s['key'], round(s['cost_bps_each_way'])): s for s in original['summary']}
    selected = [
        ('Original close / next-open rule', 'at0_addsnone_targetNone_stopNone'),
        ('No add; $1,500 total-loss trigger', 'at0_addsnone_targetNone_stop1500'),
        ('No add; recovery profit exit + $1,500 trigger', 'at1000_adds0_target1000_stop1500'),
        ('Add $5K at −5%; keep scheduled exit; $2K trigger', 'at500_adds5000_targetNone_stop2000'),
        ('Same partial add; exit at combined break-even', 'at500_adds5000_target0_stop2000'),
        ('Add $10K at −5%; break-even; no loss trigger', 'at500_adds10000_target0_stopNone'),
    ]
    candidate = lookup[(selected[3][1], 11)]
    assert candidate['pnl'] == 3884.01 and candidate['pnl_on_10k_reserved_budget'] == 2589.34
    comparison = table(['Policy', 'Maximum planned cash', 'Modeled net P&L', 'P&L scaled to $10K maximum', 'Worst trade on $10K maximum'], [
        [label, usd(s['budget']), money(s['pnl']), money(s['pnl_on_10k_reserved_budget']), money(s['worst_trade'] * 10000 / s['budget'])]
        for label, key in selected for s in [lookup[(key, 11)]]
    ])
    costs = table(['Policy', 'Before costs', '0.11% each way', '0.26% each way'], [
        [label, *[money(lookup[(key, bps)]['pnl_on_10k_reserved_budget']) for bps in [0, 11, 26]]]
        for label, key in selected
    ])
    nearby = table(['Add after price falls', '$1,500 loss trigger', '$2,000 loss trigger', '$3,000 loss trigger'], [
        [f'{trigger / 100:.0f}%', *[money(lookup[(f'at{trigger}_adds5000_targetNone_stop{cap}', 11)]['pnl_on_10k_reserved_budget']) for cap in [1500, 2000, 3000]]]
        for trigger in [300, 400, 500, 600, 700]
    ])
    folds = table(['Later dates tested', 'Best earlier score', 'Best earlier neighborhood', 'Original strategy'], [
        [f"{f['test_dates'][0]} → {f['test_dates'][1]}", money(f['raw_selection']['test_pnl_per_10k']),
         money(f['neighbor_selection']['test_pnl_per_10k']), money(f['baseline_test_pnl_per_10k'])]
        for f in analysis['chronological_folds']
    ] + [['Total across 24 later trades', *[money(analysis['chronological_totals_per_10k'][k]) for k in ['raw_search', 'neighbor_search', 'baseline']]]])
    historical_names = [('original', 'Original scheduled exits'), ('partial_add_keep_exit', 'Partial add; scheduled exits'),
                        ('partial_add_break_even', 'Partial add; break-even'), ('double_break_even_no_stop', 'Double; break-even; no loss trigger')]
    historical = table(['Fixed policy', 'All 433: mean P&L', '2025 onward, 217: mean P&L', 'All 433: 0.26% cost'], [
        [label, *[money(next(s['mean_per_10k_reserved'] for s in stress['summaries'] if s['policy'] == key and s['split'] == split and s['cost_bps_each_way'] == bps))
                  for split, bps in [('all', 11), ('holdout', 11), ('all', 26)]]]
        for key, label in historical_names
    ])
    family = table(['Best in this family, selected after seeing all dates', 'P&L per $10K reserved'], [
        ['Keep original strategy', money(analysis['baseline']['normalized_total_by_cost'][1])],
        ['Partial / repeated adds with any tested exit', money(analysis['best_by_cost']['11']['normalized_total_by_cost'][1])],
        ['Add, then exit at combined break-even', money(analysis['best_break_even_after_add'][0]['normalized_total_by_cost'][1])],
        ['Add, then accept a small loss on recovery', money(analysis['best_accept_small_loss_after_add'][0]['normalized_total_by_cost'][1])],
    ])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.parse_math': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    triggers = [300, 400, 500, 600, 700]
    for cap, color in [(1500, '#b06b7a'), (2000, '#6650bd'), (3000, '#478782')]:
        values = [lookup[(f'at{trigger}_adds5000_targetNone_stop{cap}', 11)]['pnl_on_10k_reserved_budget'] for trigger in triggers]
        axes[0].plot([t / 100 for t in triggers], values, marker='o', color=color, label=f'${cap:,} loss trigger')
    axes[0].axhline(1911.82, color='#687586', linestyle='--', linewidth=1.1, label='Original rule')
    axes[0].set_title('Partial-add results depend on the loss limit', loc='left', fontsize=12)
    axes[0].set_xlabel('Price decline that triggers the addition (%)')
    axes[0].set_ylabel('36-trade P&L per $10K maximum capital ($)')
    axes[0].set_ylim(0, 3200)
    axes[0].set_xticks([3, 4, 5, 6, 7])
    axes[0].grid(axis='y', alpha=.14)
    axes[0].legend(frameon=False, loc='upper left', ncol=2, fontsize=9)
    values = [analysis['chronological_totals_per_10k'][k] for k in ['raw_search', 'neighbor_search', 'baseline']]
    axes[1].bar([0, 1, 2], values, color=['#b06b7a', '#8c7ccd', '#478782'], width=.62)
    axes[1].set_xticks([0, 1, 2], ['Choose best\nearlier score', 'Choose stable\nearlier region', 'Keep original\nstrategy'])
    axes[1].set_title('Earlier-date choices lag on later dates', loc='left', fontsize=12)
    axes[1].set_ylabel('24 later trades: P&L per $10K maximum ($)')
    axes[1].set_ylim(-850, 6250)
    axes[1].axhline(0, color='#687586', linewidth=.8)
    axes[1].grid(axis='y', alpha=.14)
    for i, value in enumerate(values):
        axes[1].text(i, value + (110 if value >= 0 else -110), usd(value), ha='center', va='bottom' if value >= 0 else 'top', fontsize=11)
    fig.suptitle('Averaging-down results and chronological checks', x=.065, ha='left', fontsize=17, fontweight='bold')
    fig.text(.065, .025, 'Fixed Rank 1 selections · assumed 0.11% cost each way · all strategies scaled to the same maximum reserved cash\n'
             'Left: $10K initial + $5K add before scaling. Right: retrospective chronological checks; no untouched holdout.', color='#536078', fontsize=9)
    fig.tight_layout(rect=[.01, .12, .99, .91], w_pad=3)
    fig.savefig(ROOT / 'average_down_grid.png', dpi=180, bbox_inches='tight')
    fig.savefig(ROOT / 'average_down_grid.svg', bbox_inches='tight')
    plt.close(fig)
    svg = (ROOT / 'average_down_grid.svg').read_text()
    svg = svg[svg.index('<svg'):]
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rank 1 · searching for an averaging-down sweet spot</title>
<style>
body{{margin:0;background:#f4f6fa;color:#203047;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:38px 24px 65px}}
h1{{font-size:36px;line-height:1.2;max-width:1000px}}h2{{font-size:23px;margin-top:38px}}p{{max-width:1070px}}.lead{{font-size:20px}}
.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#6551a5;font-weight:700}}.callout{{background:#eae6f8;border-left:4px solid #7b68bd;padding:16px 22px;border-radius:5px}}
.note{{color:#536078}}.scroll{{overflow:auto;background:white;border:1px solid #dce1eb;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}
th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid #e7ebf2}}th{{background:#eef1f7}}th:first-child,td:first-child{{text-align:left}}
.gain{{color:#187a61;font-weight:600}}.loss{{color:#a4435b;font-weight:600}}a{{color:#6250a4}}li{{margin-bottom:8px}}svg{{width:100%;height:auto;background:white;border-radius:12px}}
.formula{{font:16px/1.8 ui-monospace,monospace;background:white;padding:16px 22px;border-radius:10px;overflow:auto}}details{{margin-top:24px}}summary{{cursor:pointer;font-weight:650}}
footer{{border-top:1px solid #dce1eb;margin-top:36px;padding-top:18px;font-size:13px;color:#536078}}code{{font-size:13px}}
</style></head><body><main>
<div class="eyebrow">Theoretical research · November 7, 2026 · fixed historical selections</div>
<h1>A partial addition looks promising; an optimal rule is not established</h1>
<p class="lead">I tested 7,058 add/exit configurations on the 36 corrected Rank 1 positions. The highest full-sample result after assumed costs uses one half-size addition after a 5% price decline, a $2,000 combined loss trigger, and the original exit schedule. Selling immediately at combined break-even generally sacrifices too much of the recovery in this Rank sample.</p>
<div class="callout"><strong>Research candidate:</strong> $10K initial purchase → add $5K once at a 5% decline → monitor a $2K total-position loss trigger → otherwise retain the original close/next-open exit. This is a rule to test prospectively, not a recommendation to commit real capital.</div>
<h2 id="candidate">The candidate in concrete numbers</h2>
<ol>
<li>Buy $10,000 at the hypothetical opening entry price.</li>
<li>If the stock falls 5% from that price, add $5,000 once. A further decline does not permit more additions.</li>
<li>Monitor a $2,000 loss across the complete position, including modeled costs. The trigger is active from the first purchase and does not reset when shares are added.</li>
<li>Do not sell merely because the combined position recovers to break-even. Retain the original schedule: close that day if the stock is flat/up from the initial entry, otherwise the next session’s open.</li>
<li>The deadline remains that close or next open. This grid does not test holding for two weeks or waiting indefinitely for a rebound.</li>
</ol>
<div class="formula">Example before costs: initial price $100; add at $95<br>
Shares = 100 + 5,000 / 95 = 152.6316<br>
Combined break-even = $15,000 / 152.6316 = $98.2759<br>
Rebound needed from the add = 3.45%<br>
$2,000 total-loss trigger ≈ $85.1724, or 14.83% below the initial price</div>
<p>At the instant of the add, the existing $500 loss remains. The addition changes future exposure and the recovery price. It does not remove that loss. The stop is wide in percentage terms; there is no claim that this particular tolerance fits the user’s account.</p>
<h2 id="comparison">Compare equal capital, not just larger dollar profits</h2>
{comparison}
<p class="note">All rows cover the same 36 positions, September 18–November 6, with assumed costs of 0.11% on each purchase and sale. The partial-add rule reserves $15K; scaling it to $10K means $6,666.67 initially plus $3,333.33 reserved, with its total-loss trigger scaled to $1,333.33. Unused reserves earn zero. These are fixed-size, un-compounded hypothetical trades, not a fully funded account simulation.</p>
<p>The best no-add recovery variant arms a $1,000 profit exit after the original position first falls $1,000, while retaining the $1,500 loss trigger and scheduled deadline. It comes within $57.65 of the best partial-add result on equal capital. The simpler $1,500 stop-only control also improves on the baseline here.</p>
<h2 id="break-even">Does break-even—or accepting a small recovery loss—help?</h2>
{family}
<p>The table selects the best result within each family after inspecting all dates. Even with this selection advantage, neither the break-even family nor the small-recovery-loss family beats the original strategy at the primary cost assumption. A recovery exit at −$250 or −$500 is distinct from an immediate hard loss trigger at that amount.</p>
<h2 id="costs">Costs can change which orders fire</h2>
{costs}
<p class="note">Every number above is scaled to a $10K maximum cash budget. Costs are execution assumptions, not measured broker charges. The replay recalculates break-even and stop prices when costs change; results are not produced by subtracting a flat fee afterward. Occasionally a higher-cost path stops earlier and avoids a later gap, so a policy’s result need not move monotonically.</p>
<h2 id="neighbors">Nearby settings do not establish a stable optimum</h2>
<p>These are one $5K addition, the scheduled exit, and different add/stop combinations. Amounts describe the unscaled $10K initial position; results are scaled to the same $10K maximum reserve.</p>
{nearby}
<p>The strongest coarse-grid region is roughly a 3–5% initial decline with a half-size addition and the $2,000 total-loss trigger. Tightening that trigger to $1,500 cuts off recoveries; widening it to $3,000 lets larger failures through. This neighborhood was identified retrospectively and is not a stable result across all nearby choices.</p>
<h2 id="validation">Choosing the rule before the next trades is the harder test</h2>
{folds}
<p>The first selection uses 11 earlier trades, skips the immediately preceding date to avoid next-open timing leakage, then evaluates the next six. The training window expands for each subsequent block. “Best earlier neighborhood” selects using the median result of adjacent trigger, profit-target and loss-limit settings. The selected rules change substantially across blocks.</p>
<p><strong>The two tested selection procedures both underperform the original rule on the later dates.</strong> These are retrospective diagnostics, not pristine out-of-sample evidence: the dates and many alternative strategies had already been inspected during the research.</p>
<div role="img" aria-label="Partial-add performance varies with loss limits, and selecting strategies on earlier dates underperforms the original rule on the next dates.">{svg}</div>
<h2 id="drivers">One trade accounts for more than the entire apparent advantage</h2>
<p>The partial-add candidate improves on the original by $677.52 after scaling to equal capital. TDTH alone contributes $769.62 of that improvement. Removing the same TDTH observation from both strategies leaves the candidate <strong>$92.10 behind</strong>. This is a sensitivity check, not a reason to exclude TDTH from the headline.</p>
<p>TDTH is also thinly traded: its first available entry bar contains 200 shares, while a $10K entry needs about 5,348 shares at the recorded price. The aggregate does not establish that the modeled order could have filled there. That is material because so much of the ranking advantage depends on this case.</p>
<h2 id="historical">A separate check across 433 historical dip episodes</h2>
{historical}
<p class="note">Average P&L per episode per $10K maximum reserved capital, not total portfolio profit. The 433 retained episodes span January 2022–October 2026 and come from a fixed 480-event sample after prior coverage checks. Each follows a prior-day decline of at least 5% in a liquid stock. These are not Rank model picks. The policies were fixed for this follow-up check; the source sample had been used in previous research.</p>
<p>The broad sample does not establish a general “fall, add, recover” advantage. Mean results are only a few dollars after the primary assumed costs, and higher costs turn the displayed candidates negative. Different entry signals, current-universe survivorship and minute-based closing prices limit direct comparison with Rank picks.</p>
<h2 id="scope">What was tested</h2>
<ul>
<li>Initial price-decline triggers: 2%, 2.5%, 3%, 4%, 5%, 6%, 7%, 8%, 10%, 12%.</li>
<li>No add; single additions of 25%, 50%, 100%, or 200% of the initial stake; and two-add patterns of 50% + 50%, 100% + 100%, or 100% + 200%.</li>
<li>Second additions occur after another multiplicative decline of the same percentage. A 5% ladder therefore uses 95% and 90.25% of the original price.</li>
<li>After activation, combined recovery exits at −$500, −$250, $0, +$250, +$500, +$1,000, +$2,000, or the scheduled exit.</li>
<li>Total-loss triggers of $500, $1,000, $1,500, $2,000, $3,000, $5,000, or none. Rules that would exit immediately at or before the add are excluded.</li>
<li>Same-day close versus the original close/next-open deadline; costs of 0%, 0.11%, and 0.26% each way.</li>
<li>7,058 configurations × 36 positions × 3 cost assumptions × 2 OHLC paths = 1,524,528 individual replays. Many configurations produce the same decisions; there are 4,730 distinct net-0.11% daily return vectors when rounded to cents.</li>
</ul>
<details><summary>Execution assumptions, limits and reproducibility</summary>
<p>Initial Rank entries use the repaired dashboard’s opening-price assumption. Some picks were published after the open, so this is not proof of achievable live performance. Choosing a same-close exit from that exact closing price is also idealized; a live implementation needs a decision price available before the applicable order cutoff. Adds and loss triggers are evaluated in regular hours, including observed opening gaps. No additions occur at the scheduled closing auction or next opening liquidation. No orders or production settings were changed.</p>
<p>Minute bars are expanded into O-H-L-C and O-L-H-C paths; the lower result is retained for each position. Nine previously identified ambiguous minutes were replaced with second bars whose OHLC matches the minute. The two paths are sensitivities, not guaranteed lower/upper bounds on all tick sequences. Bar-high/low touches do not establish executable size, order priority, spreads or market impact. Fractional shares are allowed; taxes, financing, interest on cash and simultaneous portfolio funding are not modeled.</p>
<p>A loss trigger is not a guaranteed maximum loss. Observed gaps fill at the available bar opening price and can exceed the trigger. <a href="https://www.finra.org/investors/insights/stop-orders-factors-consider-during-volatile-markets">FINRA explains stop execution and temporary price movements</a>. <a href="https://massive.com/docs/rest/stocks/aggregates/custom-bars">Massive documents how qualifying trades form aggregate bars</a>.</p>
<p>Verification passed: six synthetic gap/order/cost checks, {verification['independent_reference_and_cashflow_checks']:,} comparisons with the independent one-add replay plus cashflow reconciliation, and {verification['saved_summary_reconciliations']:,} saved-summary reconciliations. Array checks cover finite values, capital limits, lower/upper ordering and adverse marks. Closed-trade drawdown is not intraday portfolio drawdown.</p>
</details>
<footer>Reproduce: <code>search_average_down.py --deadline original</code>, <code>--deadline close</code>, <code>analyze_average_down_grid.py</code>, <code>stress_average_down.py</code>, <code>verify_average_down_search.py</code>, then <code>render_average_down_grid.py</code>.
<p><a href="average_down_grid_analysis.json">Search and chronological results</a> · <a href="average_down_historical_stress.json">Historical stress results</a> · <a href="average_down_grid_verification.json">Verification and hashes</a> · <a href="average_down_grid.png">Export figure</a></p></footer>
</main></body></html>
'''
    (ROOT / 'average_down_grid_report.html').write_text(html)
    print('Wrote average_down_grid_report.html, average_down_grid.png and average_down_grid.svg')


if __name__ == '__main__':
    main()
