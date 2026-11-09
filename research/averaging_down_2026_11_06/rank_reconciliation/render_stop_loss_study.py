"""Standalone report and recovery chart for the fixed-dollar stop replay."""
import gzip
from html import escape
import json
import os
from pathlib import Path
import sys

from rank1_stop_loss_study import local_time, regular_bars

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/private/tmp/stock-picker-audit-plotting')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/stock-picker-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = json.loads((ROOT / 'rank1_stop_loss_study.json').read_text())
base = data['baseline']
limits = [500, 600, 700, 1000]
usd = lambda value: f'−${abs(value):,.2f}' if value < 0 else f'${value:,.2f}'
money = lambda value: f'<span class="{"loss" if value < 0 else "gain"}">{usd(value)}</span>'

alhc = next(row for row in data['rows'] if row['ticker'] == 'ALHC')
with gzip.open(next((ROOT / 'minute_bars').glob('ALHC_2026-10-09_*')), 'rt') as handle:
    bars = regular_bars(json.load(handle)['bars'], '2026-10-09')
x = [local_time(bar).hour * 60 + local_time(bar).minute - 570 + 0.5 for bar in bars]
pnl = lambda price: 10000 * (price / alhc['entry_price'] - 1)
closes = [pnl(bar['c']) for bar in bars]
lows = [pnl(bar['l']) for bar in bars]
highs = [pnl(bar['h']) for bar in bars]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1.5]})
for ax in axes:
    ax.fill_between(x, lows, highs, color='#a99ef0', alpha=.30, label='Minute high / low')
    ax.plot(x, closes, color='#6350b9', linewidth=1.6, label='Minute close')
    ax.scatter([0], [0], color='#26344d', zorder=5, s=25)
    ax.axhline(0, color='#64748b', linewidth=.8)
    for limit, color in zip(limits, ['#d5a3a9', '#c67c88', '#b3566a', '#95364f']):
        ax.axhline(-limit, color=color, linewidth=.9, linestyle='--')
    ax.set_ylim(-1550, 1300)
    ax.grid(axis='y', alpha=.13)
    ax.set_ylabel('Position P&L ($)')
axes[0].set_xlim(-.3, 20)
axes[0].set_xticks([0, 5, 10, 15, 20], ['9:30', '9:35', '9:40', '9:45', '9:50'])
axes[0].set_title('The opening 20 minutes', loc='left', fontsize=12)
axes[0].text(2, -1410, 'Low: −$1,333.81', fontsize=10, color='#95364f')
for limit in limits:
    axes[0].text(19.6, -limit + 25, f'−${limit:,}', ha='right', color='#95364f', fontsize=9)
axes[1].set_xlim(0, 390)
axes[1].set_xticks([0, 90, 210, 330, 390], ['9:30', '11:00', '13:00', '15:00', '16:00'])
axes[1].set_title('The same position through the close', loc='left', fontsize=12)
axes[1].scatter([390], [alhc['original_pnl']], color='#187a61', zorder=5, s=35, clip_on=False)
axes[1].text(375, 1050, 'Closing result: +$879.60', ha='right', color='#187a61', fontweight='bold')
axes[1].legend(loc='lower right', frameon=False)
fig.suptitle('ALHC crossed every proposed stop, then finished profitable', x=.065,
             ha='left', fontsize=16, fontweight='bold')
fig.text(.065, .025, 'Oct 9, 2026 · $10,000 hypothetical opening entry at $6.935 · Eastern time · before costs\n'
         'Minute ranges show traded prices, not guaranteed stop fills. Opening-price entry is the dashboard assumption.', fontsize=9, color='#536078', parse_math=False)
fig.tight_layout(rect=[.01, .10, .99, .91], w_pad=2)
fig.savefig(ROOT / 'rank1_stop_recovery.png', dpi=180, bbox_inches='tight')
fig.savefig(ROOT / 'rank1_stop_recovery.svg', bbox_inches='tight')
plt.close(fig)
svg = (ROOT / 'rank1_stop_recovery.svg').read_text()
svg = svg[svg.index('<svg'):]


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


overview = table(['Loss trigger', 'Trades stopped', 'Winners stopped', 'Total P&L', 'Change', 'Worst trade', 'Closed-trade drawdown'], [
    ['Existing exit rule', '—', '—', money(base['total_pnl']), '—', money(base['worst_trade']), usd(-base['closed_trade_max_drawdown'])],
    *[[f"${row['loss_limit']:,} ({row['entry_price_drop_pct']:.0f}%)", row['stop_triggers'], row['winner_stops'],
       money(row['total_pnl']), money(row['change_vs_original']), money(row['worst_trade']), usd(-row['closed_trade_max_drawdown'])]
      for row in data['comparisons']],
])
recovery = table(['Date', 'Ticker', 'Deepest intraday loss', 'Original final P&L', 'Limits that stop it'], [
    [row['date'], row['ticker'], usd(-row['maximum_intraday_loss']), money(row['original_pnl']),
     ', '.join(f'${other["loss_limit"]:,}' for other in data['rows'] if other['date'] == row['date'] and other['winner_stopped'])]
    for row in data['rows'] if row['loss_limit'] == 500 and row['winner_stopped']
])
effects = table(['Loss trigger', 'Trades improved', 'Trades worsened', 'Savings', 'Cost of premature exits', 'Net change'], [
    [f"${row['loss_limit']:,}", row['improved_trades'], row['worsened_trades'],
     money(row['saved_on_improved_trades']), money(-row['lost_on_worsened_trades']), money(row['change_vs_original'])]
    for row in data['comparisons']
])
ledger = []
for row in data['rows']:
    if row['loss_limit'] != 500:
        continue
    related = [other for other in data['rows'] if other['date'] == row['date']]
    ledger.append([row['date'], row['ticker'], usd(-row['maximum_intraday_loss']), money(row['original_pnl']),
                   *[money(next(other['modeled_pnl'] for other in related if other['loss_limit'] == limit)) for limit in limits]])
all_trades = table(['Date', 'Ticker', 'Intraday low P&L', 'Existing exit', '$500 stop', '$600 stop', '$700 stop', '$1,000 stop'], ledger)
assumptions = ''.join(f'<li>{escape(item)}</li>' for item in data['assumptions'])
html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rank 1 · fixed loss limits</title><style>
body{{margin:0;background:#f4f6fa;color:#1f2d43;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1190px;margin:auto;padding:40px 24px 65px}}
h1{{font-size:34px;line-height:1.15}}h2{{margin-top:38px;font-size:23px}}.eyebrow{{text-transform:uppercase;font-size:12px;letter-spacing:.15em;color:#6756aa;font-weight:700}}
.lead{{font-size:20px;max-width:950px}}.note{{color:#536078}}.callout{{background:#eae6f8;border-left:4px solid #7968ba;padding:18px 22px;border-radius:5px}}
.scroll{{overflow:auto;background:white;border:1px solid #dce1eb;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}
th,td{{padding:12px 14px;text-align:right;border-bottom:1px solid #e7ebf2}}th{{background:#eef1f7}}th:first-child,td:first-child{{text-align:left}}
.gain{{color:#187a61;font-weight:600}}.loss{{color:#a4435b;font-weight:600}}svg{{width:100%;height:auto;background:white;border-radius:12px;margin:24px 0 0}}a{{color:#6250a4}}li{{margin-bottom:7px}}
details{{margin-top:25px}}summary{{cursor:pointer;font-weight:650}}code{{font-size:13px}}footer{{margin-top:32px;border-top:1px solid #dce1eb;padding-top:20px;font-size:13px;color:#536078}}
</style><main><div class="eyebrow">Historical replay · fixed selections · November 6, 2026 snapshot</div>
<h1>What if each Rank 1 position had a dollar loss limit?</h1>
<p class="lead">Every proposed limit stopped an eventual winner. In these 36 positions, $600 produced the highest profit among the four stop rules, but all four reduced profit versus the existing close / next-open exits.</p>
<p class="note">Same $10,000 position size and opening entries as the repaired dashboard, September 18–November 6, 2026. Sell at the close when flat or up; otherwise sell at the next session’s open. No re-entry. All results are gross hypothetical P&amp;L.</p>
{overview}
<p class="note">Drawdown measures the decline from a prior peak after scheduled trade exits. It is not maximum intraday account drawdown. A trigger is not a guaranteed loss cap: the $500 rule still loses $521.42 on TSM because it gaps past the threshold at the next opening sale.</p>
<h2>Recoveries the stops would have missed</h2>{recovery}{svg}
<div class="callout">ALHC drops below every proposed trigger in the opening two minutes, then closes up $879.60. TEM also drops just beyond a $1,000 loss and recovers to a $194.20 loss by the scheduled exit. Recovery matters even when a position never becomes a winner.</div>
<h2>The losses prevented versus recoveries forfeited</h2>{effects}
<p>The $500 rule saves $2,921.91 on XENE, TDTH and CTVA, but costs $5,223.53 on eight other positions. Its net effect is −$2,301.62. The ticker referred to as “RDTGH” is TDTH in this dataset.</p>
<p>The $600 result benefits from narrowly avoiding the FSLY, TRLV and FRMI dips of $560–$594. It still stops ALHC and four positions that later recover to smaller losses. Choosing a threshold after observing these paths can overfit this short sample.</p>
<h2>Execution limits matter</h2><p>These use optimistic fills at the trigger unless an observed minute opens below it. No such minute-open gap changes the modeled intraday stop fills here. Intraminute gaps, spreads and insufficient depth remain unknown. TDTH’s first $500-stop minute has only 1,031 traded shares against a modeled position of about 5,348 shares; the full position cannot be assumed to fill at the trigger from that bar alone. QH also trades sparsely. Opening-price entries can precede the publication of the picks.</p>
<p>A stop becomes a market order, and the execution price can differ from the trigger. See <a href="https://www.finra.org/investors/insights/stop-orders-factors-consider-during-volatile-markets">FINRA’s stop-order guidance</a>. The <a href="https://massive.com/docs/rest/stocks/aggregates/custom-bars">Massive aggregate documentation</a> explains that qualifying trades produce each bar; intervals with no qualifying trades may have no bar.</p>
<details><summary>All 36 positions</summary>{all_trades}</details>
<details><summary>Exact assumptions and verification</summary><ul>{assumptions}</ul>
<p>All 36 minute-series session lows match the separate daily-price snapshot. An independent daily-low calculation agrees with all 144 threshold decisions and all four ideal-fill totals. First-trigger and gap behavior are checked with synthetic cases.</p></details>
<footer>Reproduction: <code>rank1_stop_loss_study.py</code> and <code>render_stop_loss_study.py</code>. Complete results, timestamps, input hashes and coverage: <a href="rank1_stop_loss_study.json">rank1_stop_loss_study.json</a>. No dashboard settings or trading orders were changed.</footer></main></html>'''
(ROOT / 'rank1_stop_loss_report.html').write_text(html)
print('Wrote rank1_stop_loss_report.html and rank1_stop_recovery.png/.svg')
