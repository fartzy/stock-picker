"""Render the frozen research results as a self-contained HTML report (stdlib only)."""
from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name: str):
    return json.loads((HERE / name).read_text())


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pct(value, digits=3) -> str:
    return "—" if value is None else f"{value * 100:+.{digits}f}%"


LABELS = {
    "all_eligible": "All eligible stocks",
    "shock_2pct": "Two prior drops, each ≥2%",
    "shock_2pct_new252low": "Two ≥2% drops + yesterday made a 252-session low",
    "shock_2pct_close_near252low": "Two ≥2% drops + prior close within 2% of low",
    "shock_2pct_nearlow_open_below_low": "Above near-low setup → opens below prior low",
    "shock_2pct_nearlow_open_0_to_1pct_above_low": "Above near-low setup → opens 0–1% above low",
    "shock_2pct_nearlow_open_gt1pct_above_low": "Above near-low setup → opens >1% above low",
    "shock_2pct_recovery_negative": "Two ≥2% drops → gaps down again",
    "shock_2pct_recovery_0_to_25pct": "Two ≥2% drops → recovers 0–25% of closing loss at open",
    "shock_2pct_recovery_gt25pct": "Two ≥2% drops → recovers >25% of closing loss at open",
    "all_eligible_close_near252high": "Prior close within 2% of 252-session high",
    "secondary_shock_1pct": "Sensitivity: two prior drops, each ≥1%",
    "secondary_shock_1pct_close_near252low": "Sensitivity: two ≥1% drops + near low",
    "secondary_shock_3pct": "Sensitivity: two prior drops, each ≥3%",
    "secondary_shock_3pct_close_near252low": "Sensitivity: two ≥3% drops + near low",
    "all_eligible_close_near252low": "Prior close within 2% of 252-session low",
    "all_eligible_new252low": "Yesterday made a new/tied 252-session low",
}


def main() -> None:
    results = read("results.json")["summaries"]
    protocol = read("protocol.json")
    case = read("mrcy_case.json")
    plan = read("implementation_plan.json")
    triggers = read("trigger_candidates.json")
    sources = read("primary_source_index.json")["sources"]
    manifest = read("run_manifest.json")
    overall = {r["cohort"]: r for r in results if r["era"] == "all"}
    baseline = overall["all_eligible"]
    near = overall["shock_2pct_close_near252low"]
    gap_down = overall["shock_2pct_recovery_negative"]
    near_ci = near["matched_excess_bootstrap"]["ci95"]
    gap_ci = gap_down["matched_excess_bootstrap"]["ci95"]
    shock_eras = {r["era"]: r for r in results if r["cohort"] == "shock_2pct"}
    selected = ["shock_2pct", "shock_2pct_new252low", "shock_2pct_close_near252low",
                "shock_2pct_nearlow_open_0_to_1pct_above_low", "shock_2pct_recovery_negative"]
    simple_rows = "\n".join(
        f'<tr><th scope="row">{esc(LABELS[k])}</th><td>{overall[k]["rows"]:,}</td>'
        f'<td>{overall[k]["dates"]:,}</td><td>{pct(overall[k]["gross_mean_equal_date"])}</td>'
        f'<td>{pct(overall[k]["net_mean_equal_date_by_roundtrip_bps"]["25"])}</td></tr>'
        for k in selected
    )
    features = "\n".join(
        f'<article class="feature"><span class="eyebrow">Candidate {i}</span><h3>{esc(f["name"])}</h3>'
        f'<p>{esc(f["question"])}</p><p class="formula">{esc(f["formula"])}</p>'
        f'<p><strong>Available:</strong> {esc(f["timing"])}</p>'
        f'<p><strong>What is new:</strong> {esc(f["novelty"])}</p>'
        f'<p class="muted">{esc(f["guard"])}</p></article>'
        for i, f in enumerate(plan["priority_features"], 1)
    )
    evidence = "\n".join(
        f'<tr><th scope="row"><a href="{esc(s["url"])}">{esc(s["title"])} ({s["year"]})</a></th>'
        f'<td>{esc(s["sample"])}<br><br>{esc(s["horizon"])}</td>'
        f'<td>{esc(" ".join(s["findings"]))}<br><br><span class="muted">{esc(" ".join(s["limitations"][:2]))}</span></td></tr>'
        for s in sources[:5]
    )
    phases = "\n".join(
        f'<details><summary>Phase {p["phase"]}: {esc(p["name"])}</summary><p>{esc(p["work"])}</p>'
        f'<p><strong>Verify:</strong> {esc(p["verification"])}</p><p><strong>Guard:</strong> {esc(p["guards"])}</p>'
        f'<ul>{"".join(f"<li><code>{esc(x)}</code></li>" for x in p.get("files_to_review", p.get("files_to_touch_if_approved", p.get("references", []))))}</ul></details>'
        for p in plan["phases"]
    )
    trigger_rows = "\n".join(
        f'<tr><th scope="row">{esc(t["name"])}</th><td><code>{esc(t["flag"])}</code><p>{esc(t["definitions"])}</p></td>'
        f'<td>{esc(t["purpose"])}<p class="muted">{esc(t["decision_time"])}</p></td></tr>'
        for t in triggers["candidates"]
    )
    payload = json.dumps({"rows": results, "labels": LABELS}, allow_nan=False).replace("<", "\\u003c")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Morning dip patterns · November 9, 2026</title>
<style>
:root{{color-scheme:light;--ink:#182b3a;--muted:#546779;--accent:#086f72;--line:#d9e3e8;--paper:#fff;--bg:#edf3f5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:auto;padding:48px 28px 70px}}header{{padding:0 0 30px}}h1{{font-size:clamp(2.1rem,5vw,3.7rem);line-height:1.08;max-width:850px;letter-spacing:-.04em;margin:15px 0 20px}}
h2{{font-size:1.65rem;line-height:1.2;margin:0 0 18px}}h3{{line-height:1.3;margin:8px 0 15px}}p{{max-width:94ch;margin:12px 0}}a{{color:var(--accent);text-underline-offset:3px}}a:hover{{color:#004d53}}
.eyebrow{{text-transform:uppercase;letter-spacing:.11em;font-weight:750;font-size:.76rem;color:var(--accent)}}.lead{{font-size:1.2rem;max-width:830px}}.muted,small{{color:var(--muted)}}
section{{background:var(--paper);padding:28px;margin:20px 0;border:1px solid var(--line);border-radius:14px}}.finding{{border-left:5px solid var(--accent)}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}}
.metric{{padding:18px;background:#e3f0ee;border-radius:10px}}.metric strong{{display:block;font-size:1.8rem;line-height:1.25}}.metric span{{font-size:.88rem}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:.87rem}}th,td{{padding:12px 13px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;font-variant-numeric:tabular-nums}}
thead th{{background:#edf4f5;font-size:.76rem;letter-spacing:.02em;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}tbody th{{font-weight:550;min-width:245px}}tbody tr:last-child>*{{border-bottom:none}}tbody tr:hover{{background:#f6fafb}}
.sources td,.sources th{{text-align:left;min-width:220px}}.sources td:nth-child(3){{min-width:320px}}.controls{{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}}label{{font-weight:650;font-size:.9rem}}select{{display:block;font:inherit;color:var(--ink);padding:8px 34px 8px 10px;margin-top:6px;border:1px solid #a3b8c5;border-radius:6px;background:white}}
.features{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}}.feature{{border:1px solid var(--line);border-radius:10px;padding:23px}}.feature p{{font-size:.93rem}}.formula,code{{font-family:ui-monospace,monospace;font-size:.82rem;overflow-wrap:anywhere}}.formula{{padding:12px;background:#f1f5f7;border-radius:6px}}
.note{{padding:15px 18px;background:#fff7e9;border-left:3px solid #b57717;border-radius:4px}}details{{border-top:1px solid var(--line);padding:15px 0}}summary{{cursor:pointer;font-weight:650}}details ul{{font-size:.9rem}}.negative{{color:#a52d37}}.positive{{color:#08634f}}.badge{{display:inline-block;font-size:.7rem;color:var(--muted);background:#edf2f6;border-radius:4px;padding:1px 5px;margin:3px 0}}
nav{{display:flex;gap:17px;flex-wrap:wrap;font-size:.9rem;margin-top:20px}}footer{{font-size:.85rem;color:var(--muted)}}.report-data td{{white-space:nowrap}}.report-data th{{max-width:340px}}:focus-visible{{outline:3px solid #259a9d;outline-offset:3px}}
@media(max-width:700px){{main{{padding:25px 13px}}section{{padding:20px 16px}}.features{{grid-template-columns:1fr}}.metrics{{gap:8px}}.metric{{padding:12px}}.metric strong{{font-size:1.35rem}}.metric span{{font-size:.73rem}}}}
@media print{{body{{background:#fff}}main{{padding:0;max-width:none}}section{{break-inside:avoid;border-radius:0}}.controls,nav{{display:none}}.table-wrap{{overflow:visible}}.report-data{{font-size:6pt}}.report-data th,.report-data td{{padding:4px;white-space:normal;min-width:0}}.feature{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Stock Picker research · November 9, 2026 · Same-day holding only</div>
<h1>When does a morning dip offer an edge?</h1>
<p class="lead">Track the depth, location, recovery and selling pressure of a dip as <strong>inputs alongside the existing features</strong>. The objective is to identify precise conditions that improve morning-entry to same-day-close predictions. Their added value must be tested within the model.</p>
<div class="metrics"><div class="metric"><strong>{baseline['rows']:,}</strong><span>eligible stock-days · {baseline['tickers']:,} stocks</span></div><div class="metric"><strong>{baseline['dates']:,}</strong><span>sessions · Nov 2022–Nov 2026</span></div><div class="metric"><strong>17</strong><span>descriptive cohorts · MRCY included</span></div></div>
<nav aria-label="Report sections"><a href="#triggers">Precise feature triggers</a><a href="#findings">Screen context</a><a href="#explore">Explore all setups</a><a href="#mrcy">MRCY</a><a href="#features">Measurements</a><a href="#evidence">Papers</a><a href="#method">Method & limits</a><a href="#plan">Next experiment</a></nav></header>

<section id="triggers" class="finding"><div class="eyebrow">Feature iteration · exact candidate definitions</div><h2>Keep the dip measurements and test their interactions</h2>
<p>A feature can have a weak average return on its own and still help the model when combined with trend, volume, volatility, news or other inputs. The descriptive cohort study below did not test that incremental contribution. It therefore does <strong>not settle whether these features should be added</strong>.</p>
<p>For each setup, keep both continuous measurements and a deterministic flag. Continuous values preserve the size of the effect; flags make a specific pattern reproducible and inspectable. The following cutoffs are initial hypotheses, not validated buying instructions. Existing features remain the baseline throughout.</p>
<div class="table-wrap"><table class="sources"><thead><tr><th>Candidate pattern</th><th>Exact initial flag</th><th>What it tests / timing</th></tr></thead><tbody>{trigger_rows}</tbody></table></div>
<p class="muted">Notation: t is the entry session; r1=C[t−1]/C[t−2]−1; r2=C[t−2]/C[t−3]−1. A slice [a:b] includes a and excludes b. No current-session future high, low, close or completed volume enters a morning feature. All exact definitions and validation boundaries: <a href="trigger_candidates.json">trigger_candidates.json</a>.</p>
<p><strong>How to assess predictiveness:</strong> compare the current Rank/Fit model with and without each new family on identical later-date folds, then check returns after costs, downside and stability. Threshold refinement belongs inside training dates. A rigid definition becomes a useful predictive trigger only after it shows added value on later observations.</p></section>

<section id="findings"><div class="eyebrow">Descriptive baseline · not an incremental feature test</div><h2>What the simple conditions do on their own</h2>
<p>“Two down days” means both previous close-to-close returns were at most −2%. A 52-week low means the lowest low over the <strong>252 completed sessions before entry</strong>. “Near” means yesterday closed within 2% of that low. We evaluated every eligible occurrence, including stocks never traded, with no selection from our current Rank list.</p>
<div class="table-wrap"><table><thead><tr><th>Morning setup</th><th>Stock-days</th><th>Active dates</th><th>Mean gross</th><th>After 0.25% round trip</th></tr></thead><tbody>{simple_rows}</tbody></table></div>
<p class="muted">Means give each active date equal weight after averaging its qualifying stocks. These are returns per invested session, not annualized returns or cumulative portfolio P&amp;L. The 0.25% subtraction is a cost scenario, not a measured fill cost.</p>
<p>All 17 full-period cohort means were negative after the 25-basis-point scenario. Two ≥2% down days alone averaged <strong>{pct(shock_eras['2022_2024']['gross_mean_equal_date'])}</strong> in 2022–2024, <strong>{pct(shock_eras['2025']['gross_mean_equal_date'])}</strong> in 2025 and <strong>{pct(shock_eras['2026']['gross_mean_equal_date'])}</strong> in 2026. The initial pooled positive result was not stable over time.</p>
<p>The fresh-low conjunction had <strong>{pct(overall['shock_2pct_new252low']['matched_excess_equal_date'])}</strong> excess versus same-date, similar-volatility controls. The near-low close conjunction had <strong>{pct(near['matched_excess_equal_date'])}</strong> matched excess, with a weekly-bootstrap 95% interval of <strong>{pct(near_ci[0])} to {pct(near_ci[1])}</strong>. Another gap down had {pct(gap_down['matched_excess_equal_date'])} matched excess, with an interval of {pct(gap_ci[0])} to {pct(gap_ci[1])}.</p>
<p class="note">These are standalone setup comparisons. They help describe the data and possible interactions; they do not measure how much the features add to our existing model. No proposed dip feature is rejected solely because of its standalone average.</p>
<p class="muted">MRCY's full eligible history—995 stock-days—is included in this version. It was initially excluded without a user request; that choice was corrected. All other cohort definitions and methods are unchanged. <a href="mrcy_inclusion_comparison.json">Exact before/after effect and original-version reference</a>.</p></section>

<section id="explore"><div class="eyebrow">Every fixed cohort, including failures</div><h2>Explore periods and costs</h2>
<div class="controls"><label>Period<select id="era"><option value="all">All: Nov 2022–Nov 2026</option><option value="2022_2024">Nov 2022–Dec 2024</option><option value="2025">2025</option><option value="2026">2026 through Nov 6</option></select></label><label>Round-trip cost scenario<select id="cost"><option value="0">0 bps / 0.00%</option><option value="10">10 bps / 0.10%</option><option value="25" selected>25 bps / 0.25%</option><option value="50">50 bps / 0.50%</option></select></label></div>
<p id="table-status" class="muted" aria-live="polite"></p><div class="table-wrap"><table class="report-data"><thead><tr><th>Setup</th><th>Rows / dates</th><th>Gross mean</th><th id="net-header">Net @25 bps</th><th>Matched excess</th><th>95% interval</th><th>Gross win rate</th><th>Median trade</th><th>5th pct. open→low</th><th>Winsorized mean</th></tr></thead><tbody id="results-body"></tbody></table></div>
<noscript><p>The overview table above is available without JavaScript. <a href="results.json">Download all 68 cohort-period summaries</a> to inspect detailed results.</p></noscript>
<p class="muted">Matched excess compares nonmember stocks in the same date and pre-shock volatility quintile; it is gross and does not change with the cost selector. Intervals resample whole active calendar weeks, 2,000 times, and are pointwise—not corrected for multiple comparisons. Win rate, median and adverse excursion weight stock-days equally; mean returns weight dates equally. A high win rate therefore is not a direct counterpart to the displayed mean. The open→low column describes downside during the session, not a stop-loss simulation. Winsorization clips the bottom/top 1% of row returns as a diagnostic.</p></section>

<section id="mrcy"><div class="eyebrow">Motivating example · full eligible history included in the screen</div><h2>MRCY illustrates the state we want to measure</h2>
<p>November 4 and 5 lost <strong>{pct(case['prior_close_returns'][0], 2)}</strong> and <strong>{pct(case['prior_close_returns'][1], 2)}</strong> close-to-close, a combined <strong>{pct(case['two_day_close_return'], 2)}</strong>. The November 5 low of <strong>$64.695</strong> broke the prior 252-session low of <strong>$65.04</strong> from November 21, 2025.</p>
<p>On November 6 it opened at <strong>$66.93</strong>, a <strong>+2.11%</strong> gap. That had recovered only <strong>9.34%</strong> of the two-day closing loss, so “still deeply depressed” describes the setup better than “no bounce yet.” It closed at <strong>$73.55</strong>: a hypothetical open-to-close gain of <strong>+9.89%</strong>.</p>
<p>It first fell another <strong>3.82%</strong> below that open, reaching $64.3706 in the minute beginning 09:36 ET. That later low is an outcome, not information available for an opening decision. The favorable full-session return did not imply a smooth path or identify the exact bottom.</p>
<p class="muted">This is a hypothetical opening purchase, not the realized result of the user's earlier MRCY campaign. Sources, exact prices, minute coverage and hashes: <a href="mrcy_case.json">mrcy_case.json</a>. Daily and minute caches were retrieved in the earlier study and were not refetched here.</p></section>

<section id="features"><div class="eyebrow">Four candidate families · not added to live models</div><h2>Measure the state of the dip</h2>
<p>The saved Rank model already serves 132 inputs; Fit serves 139. Both include the 19 recent price-pattern inputs and 57 open-pattern seasonalities. Returns, candle shapes, volume and volatility already cover much of this territory. Raw streak, short RSI and several dip-story features have previously been pruned. Simply adding more familiar indicators would duplicate existing information.</p>
<div class="features">{features}</div>
<p><strong>My recommended first slice:</strong> low distance / fresh-break state plus recovery fraction. They address the missing 252-session context and the user's “hasn't recovered yet” idea directly. The volume and pressure interactions come next, subject to incremental testing. No direction is hard-coded as “good”: the model should be allowed to learn that some apparently cheap setups are unfavorable.</p>
<p>Keep prior overnight losses separate from prior regular-session losses. Later research can also test stock-specific selling relative to SPY and timestamped news. A 09:35/09:45 reclaim or stabilization signal is useful to investigate separately, with a later executable entry and the same day's close as exit.</p>
<p class="muted">Audit and source locations: <a href="feature_audit.json">feature_audit.json</a>. Existing gap×volume timing: <a href="../../python/stock_picker/features/candle.py">features/candle.py:17</a>; shift contract: <a href="../../python/stock_picker/training/dataset.py">training/dataset.py:44</a>. These four full interactions have not been fit or tested for incremental model value in this cohort screen.</p></section>

<section id="evidence"><div class="eyebrow">Primary literature · horizons matter</div><h2>Reversal has mechanisms, but it can happen before we buy</h2>
<p>The strongest caution comes from Lou, Polk and Skouras: their conventional short-term reversal portfolio earns its positive component overnight and has a negative intraday component. Their monthly sorting horizon is different from our two-day rule, but it demonstrates why we must test the exact trading session.</p>
<div class="table-wrap"><table class="sources"><thead><tr><th>Primary study</th><th>Sample and horizon</th><th>Evidence and limits</th></tr></thead><tbody>{evidence}</tbody></table></div>
<p>Two additional papers were checked at publisher-abstract level only: <a href="https://doi.org/10.1017/S0022109012000270">Berkman et al. (2012)</a> on retail attention, expensive opening prices and intraday reversal; and <a href="https://doi.org/10.1111/j.1540-6261.2010.01573.x">Heston et al. (2010)</a> on half-hour return patterns and very short liquidity reversals. They motivate timing research; their unverified sample details and strategy magnitudes are not used here.</p>
<p class="muted">Five full-text primary papers had relevant sections inspected. The normal web-search tool returned 404; sources were retrieved directly from authors, universities, NBER and the New York Fed, with publisher metadata for discovery. This is a focused review, not an exhaustive or systematically complete literature search. <a href="primary_source_index.json">Source index: exact PDF pages, retrieval timestamps, hashes and access limitations</a>.</p></section>

<section id="method"><div class="eyebrow">How to interpret this screen</div><h2>Timing, controls and limitations</h2>
<ol>
<li><strong>Fixed cohort definitions; transparent inclusion correction.</strong> Seventeen cohorts, 1%/2%/3% shock thresholds, costs and temporal diagnostics were fixed before the original outcome calculation. After that screen, the user directed inclusion of MRCY; this version changes that membership only. No threshold optimization followed. The 1% and 3% rows are sensitivity checks. Cohorts overlap and are not independent strategies. The new precise feature-trigger proposals above have not been fitted or validated by this descriptive screen.</li>
<li><strong>Morning information only.</strong> Historical inputs stop at yesterday; open-conditioned rules also use the observed opening price. The label is C[t]/O[t]−1. The 252-session warmup reduces the five-year raw cache to November 1, 2022–November 6, 2026 for evaluated rows. MRCY is included; SPY and the frozen 200-name production holdout are excluded.</li>
<li><strong>Eligibility and controls.</strong> Open at least $5, previous 20-session average dollar volume at least $10m, and 252 valid completed prior bars. No future 21-day completeness filter or cooldown. Each event is compared with nonmembers in the same date/volatility quintile, with at least five controls. Volatility is measured over 20 returns ending before the two-day shock. All reported main groups have full matched-row coverage.</li>
<li><strong>Retrospective evidence.</strong> The hypotheses were motivated by a known recent example, and this history was used in earlier research. The 2025/2026 slices are temporal diagnostics, not untouched validation. The underlying registry is today's universe, excluding many dead/delisted historical names; 17 share-class symbols also lack cache coverage. No claim of a complete historical investable universe is made.</li>
<li><strong>Price basis and events.</strong> The cached daily bars are split-adjusted, exclude cash dividends and lack full five-year corporate-action certification. The historical $5 filter is on retrospectively adjusted prices. Ex-dividend dates, spinoffs and information-driven declines can mimic dips. This screen does not reproduce the live earnings/news eligibility filter, match on sector, or estimate a causal treatment effect.</li>
<li><strong>Execution remains unvalidated.</strong> The official open and close are benchmarks. <a href="https://www.nasdaqtrader.com/Trader.aspx?id=OpenClose">Nasdaq's opening cross occurs at 09:30 ET</a>; <a href="https://www.nyse.com/trade/auctions">individual NYSE securities may open later</a>. Inference: observing the final opening print cannot guarantee execution at that already established price. Open-conditioned candidates need a later-entry test, with spreads, slippage, halts and individual opening times.</li>
<li><strong>Costs and tail risk.</strong> The 0/10/25/50 bps scenarios subtract a constant total round-trip cost. They do not model size, actual spread, fees, impact or a stop. Internally valid extreme returns remain included; 1% tail clipping is shown separately. Uncertainty intervals account for weekly clustering but not all persistence, data bias or multiple testing.</li>
</ol>
<p class="note">The current production download is approximately one year. A genuine 252-session-low feature needs longer ingestion and warmup before there is adequate training coverage. Research-cache results must not be silently mixed with the production raw-price basis.</p>
<p>Primary protocol and audit trail: <a href="protocol.json">protocol.json</a> · <a href="input_snapshot.json">frozen membership and source hashes</a> · <a href="data_quality.json">quality and extreme returns</a> · <a href="run_manifest.json">run manifest</a> · <a href="daily_results.json">per-date returns</a> · <a href="verification.json">independent numerical verification</a>.</p></section>

<section id="plan"><div class="eyebrow">Proposed implementation · requires a later production decision</div><h2>Test what the new inputs add</h2>
<p>Start with the existing Rank and Fit baselines, then add each small feature family and a fixed combined candidate on identical chronological folds. Require incremental results after costs and across dates; a standalone cohort pattern cannot establish model improvement. Keep the exact same-day exit objective.</p>
{phases}
<p><a href="implementation_plan.json">Full proposed plan with files, formulas, verification checks and boundaries</a>. No production model, feature selection, live prices, trades, API or UI was changed by this research.</p></section>
<footer><p>Reproduce: <code>{esc(manifest['command'])}</code>, then <code>python3 research/morning_dip_patterns_2026_11_09/build_report.py</code>. The source pickle can be restored from its existing archive manifest if absent.</p><p>Screen protocol SHA-256: <code>{esc(manifest['protocol_sha256'])}</code>. Results are fractional values in JSON; this report displays percentages. Prepared November 9, 2026.</p></footer>
</main><script id="report-data" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent);
const era=document.getElementById('era'),cost=document.getElementById('cost'),body=document.getElementById('results-body');
const percent=(x,d=3)=>x==null?'—':(x>=0?'+':'')+(x*100).toFixed(d)+'%';
function cell(row,value,cls){{const td=document.createElement('td');td.textContent=value;if(cls)td.className=cls;row.append(td);}}
function render(){{body.replaceChildren();const rows=data.rows.filter(x=>x.era===era.value);for(const r of rows){{
 const tr=document.createElement('tr'),th=document.createElement('th');th.scope='row';th.textContent=data.labels[r.cohort];
 if(r.conditioning.startsWith('open_')){{th.append(document.createElement('br'));const b=document.createElement('span');b.className='badge';b.textContent='Uses observed open';th.append(b);}}
 if(r.sparse){{const b=document.createElement('span');b.className='badge';b.textContent='Sparse';th.append(b);}}tr.append(th);
 cell(tr,r.rows.toLocaleString()+' / '+r.dates.toLocaleString());cell(tr,percent(r.gross_mean_equal_date));
 const net=r.net_mean_equal_date_by_roundtrip_bps?.[cost.value];cell(tr,percent(net),net<0?'negative':'positive');
 cell(tr,percent(r.matched_excess_equal_date));const ci=r.matched_excess_bootstrap?.ci95;cell(tr,ci?percent(ci[0])+' to '+percent(ci[1]):'—');
 cell(tr,r.positive_row_fraction_gross==null?'—':(100*r.positive_row_fraction_gross).toFixed(1)+'%');cell(tr,percent(r.median_row_return));cell(tr,percent(r.adverse_excursion_p05_row));cell(tr,percent(r.gross_winsor01_equal_date));body.append(tr);
 }}document.getElementById('net-header').textContent='Net @'+cost.value+' bps';document.getElementById('table-status').textContent=rows.length+' fixed cohorts · '+era.options[era.selectedIndex].text+' · cost is a scenario, not a measured execution estimate';}}
era.addEventListener('change',render);cost.addEventListener('change',render);render();
</script></body></html>'''
    (HERE / "report.html").write_text(page)
    print(HERE / "report.html")


if __name__ == "__main__":
    main()
