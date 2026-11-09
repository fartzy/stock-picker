"""Render the one-add research results without changing dashboard settings."""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'rank1_average_down_study.json').read_text())
original = json.loads((ROOT / 'rank1_significance_rows.json').read_text())
usd = lambda n: f'−${abs(n):,.2f}' if n < 0 else f'${n:,.2f}'
money = lambda n: f'<span class="{"loss" if n < 0 else "gain"}">{usd(n)}</span>'


def choose(trigger, cap=None, cost=0, add=10000, be=True):
    return next(r for r in data['summary'] if r['trigger_loss'] == trigger and r['loss_cap'] == cap
                and r['cost_bps_per_side'] == cost and r['add_dollars'] == add and r['break_even_exit'] == be)


def result(row):
    return money(row['total_pnl']) if row['total_pnl'] == row['favorable_path_total_pnl'] else (
        money(row['total_pnl']) + ' to ' + money(row['favorable_path_total_pnl']))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


uncapped = table(['Original loss at add', 'Adds', 'Break-even or better exits', 'Final total P&L', 'Worst final trade', 'Lowest modeled mark'], [
    [f'${trigger:,}', row['adds'], row['break_even_exits'], result(row), money(row['worst_trade']), money(row['worst_mark'])]
    for trigger in [500, 600, 700, 1000] for row in [choose(trigger)]
])
capped = table(['Add at loss', 'Total-loss trigger', 'Gross total P&L', 'With assumed 0.11% per side', 'Worst gross trade'], [
    [f'${trigger:,}', f'${cap:,}', result(choose(trigger, cap)), result(choose(trigger, cap, 11)), money(choose(trigger, cap)['worst_trade'])]
    for trigger, cap in [(250, 500), (500, 1000), (500, 1500), (600, 1000), (600, 1500), (700, 1000), (700, 1500), (1000, 1500)]
])
controls = table(['After a $500 loss', 'Peak cash per position', 'Total gross P&L', 'Worst trade'], [
    ['Original scheduled exits', '$10,000', money(2708.87), money(-2085.56)],
    ['No add; exit on return to original break-even', '$10,000', result(choose(500, add=0)), money(choose(500, add=0)['worst_trade'])],
    ['Add $10K; exit at combined break-even', '$20,000', result(choose(500)), money(choose(500)['worst_trade'])],
    ['Add $10K; keep original scheduled exits', '$20,000', result(choose(500, be=False)), money(choose(500, be=False)['worst_trade'])],
    ['Scale preceding rule to $5K initially + $5K add', '$10,000', money(choose(500, be=False)['total_pnl'] / 2), money(choose(500, be=False)['worst_trade'] / 2)],
])
main_rows = [r for r in data['rows'] if r['policy'] == choose(500)['policy'] and r['cost_bps_per_side'] == 0 and r['added']]
details = table(['Date', 'Stock', 'Add price', 'Combined break-even', 'Exit', 'Original result', 'Add + break-even result'], [
    [r['date'], r['ticker'], f"${r['add_price']:.4f}", f"${r['combined_break_even_price']:.4f}",
     escape(r['exit_reason'].replace('_', ' ')), money(r['original_pnl']), money(r['pnl'])] for r in main_rows
])
unchanged = sum(r['pnl'] for r in original if r['date'] not in {x['date'] for x in main_rows})
failed = sum(r['pnl'] for r in main_rows if r['pnl'] < 0)
recovered = sum(r['pnl'] for r in main_rows if r['pnl'] >= 0)
assert round(unchanged + failed + recovered, 2) == choose(500)['total_pnl']
net_baseline = sum(r['net11'] for r in original)
assumptions = ''.join(f'<li>{escape(s)}</li>' for s in data['assumptions'])
html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rank 1 · one addition and a break-even exit</title><style>
body{{margin:0;background:#f4f6fa;color:#203047;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:38px 24px 65px}}
h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px;margin-top:36px}}.lead{{font-size:19px;max-width:990px}}.eyebrow{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#6551a5;font-weight:700}}
.callout{{background:#eae6f8;border-left:4px solid #7b68bd;padding:16px 22px;border-radius:5px}}.note{{color:#536078}}
.scroll{{overflow:auto;background:white;border:1px solid #dce1eb;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}
th,td{{padding:12px 14px;text-align:right;border-bottom:1px solid #e7ebf2}}th{{background:#eef1f7}}th:first-child,td:first-child{{text-align:left}}
.gain{{color:#187a61;font-weight:600}}.loss{{color:#a4435b;font-weight:600}}a{{color:#6250a4}}li{{margin-bottom:8px}}
.formula{{font:17px/1.8 ui-monospace,monospace;background:white;padding:16px 22px;border-radius:10px;overflow:auto}}
details{{margin-top:26px}}summary{{cursor:pointer;font-weight:650}}footer{{border-top:1px solid #dce1eb;margin-top:36px;padding-top:18px;font-size:13px;color:#536078}}
</style><main><div class="eyebrow">36 fixed Rank 1 picks · September 18–November 6, 2026</div>
<h1>Average down once, then exit at combined break-even</h1>
<p class="lead">A lower average purchase cost does make recovery easier. The sample also shows why a high recovery count can coexist with worse overall results: the extra shares increase the losses on failures, while a break-even exit gives up later gains on recoveries.</p>
<div class="callout">At a $500 add trigger, 8 of 11 added positions reach combined break-even or better. The full strategy makes $449.70 before costs, versus $2,708.87 with the original exit rule. The worst final loss grows from $2,085.56 to $3,754.57.</div>
<h2>The arithmetic</h2><p>“Double down” here means a second equal dollar purchase: $10,000 initially and $10,000 added once. It does not mean merely doubling the original share count.</p>
<div class="formula">$10,000 at $100 = 100 shares<br>$10,000 at $95 = 105.2632 shares<br>Combined break-even = $20,000 / 205.2632 = $97.4359<br>Required rebound from $95 = 2.5641%</div>
<p>At the instant of the second purchase, the combined position is still down $500. An immediate $500 <em>total</em>-loss exit leaves no room for recovery. With a $1,500 total-loss limit, the corresponding gross stop price is about $90.13; it is a trigger, not a guaranteed fill.</p>
<h2>One add and break-even, with the original time deadline</h2>{uncapped}
<p class="note">All 36 positions remain in each total, including picks that never trigger an addition. Exit at combined break-even when available; otherwise retain the original close/next-open deadline. No unlimited holding, repeated adds or re-entry.</p>
<h2>Adding a total-loss limit</h2>{capped}
<p class="note">Original-rule reference: {money(2708.87)} gross, or {money(net_baseline)} with the same assumed 0.11% per-side cost model. These costs are estimates, not measured fills. At cost-adjusted break-even, the target covers both purchases and estimated exit costs.</p>
<p>Among the four original add thresholds with a $1,500 cap, the $600 trigger gives the highest gross result here: $2,317.74, still below the original strategy and with up to $20K committed. A $700 add sometimes gaps through the $1,500 stop, producing a $1,577.83 loss. Adding earlier at −$250 allows a −$500 overall stop, but its cost-adjusted result is negative under the displayed execution sensitivities.</p>
<h2>Is the improvement from adding shares or changing the exit?</h2>{controls}
<p>The $5,131.63 result from adding and retaining the original exits uses more capital. Scaling that rule to a $10K maximum budget ($5K initially, $5K reserve) gives approximately $2,565.82; it does not establish an improvement over the $10K original strategy. The tested $600-trigger/$1,500-cap rule also earns slightly less with an add ($2,317.74) than the same trigger and risk cap without an add ($2,376.55).</p>
<h2>What happened to the 11 additions at −$500?</h2>{details}
<p>Seven close at modeled break-even, and FRMI gaps above its break-even target at the next open, making $2,040.06. The remaining three lose {usd(failed)} in total. The 25 unchanged positions contribute {usd(unchanged)}, leaving {usd(choose(500)['total_pnl'])} overall. These are modeled returns, not submitted trading orders.</p>
<h2>What “the model is usually not that wrong” would need to mean</h2>
<p>The relevant evidence is the distribution of returns <em>after</em> a pick drops to the add threshold. Overall ranking performance does not establish a positive conditional edge. Most recoveries can still be outweighed by a few larger failures; this sample exhibits exactly that problem for an uncapped add-and-break-even rule.</p>
<p>The choices were examined after seeing the results. None is established as an optimal live strategy. Opening entries can precede publication of the picks, thin names may not support the assumed size at a quoted price, and cost estimates do not establish actual execution. Adding also increases concentration; see <a href="https://www.finra.org/investors/insights/concentration-risk">FINRA’s explanation of concentration risk</a>. <a href="https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/types-orders">Investor.gov explains why limit orders do not guarantee execution and stops do not guarantee price</a>.</p>
<details><summary>Method, timing uncertainty and validation</summary><ul>{assumptions}</ul>
<p>Nine trigger minutes were replaced with second-level aggregates after checking their open, high, low and close against the original minute bars. Remaining differing O-H-L-C and O-L-H-C outcomes are shown as ranges; they are sensitivity scenarios, not guaranteed bounds on all possible tick paths or execution prices. Exact cashflow reconciliation passed for all 3,024 position/policy/cost cases. Synthetic checks cover initial losses surviving the add, chronology, break-even after costs, loss-limit gaps, and invalid equal-trigger/equal-stop rules.</p></details>
<footer>Reproduce with <code>rank1_average_down_study.py</code>. <a href="rank1_average_down_study.json">Complete results and source hashes</a> · <a href="rank1_average_down_minute_study.json">Original minute-only sensitivity</a>. No dashboard behavior or trading settings were changed.</footer></main></html>'''
(ROOT / 'rank1_average_down_report.html').write_text(html)
print('Wrote rank1_average_down_report.html')
