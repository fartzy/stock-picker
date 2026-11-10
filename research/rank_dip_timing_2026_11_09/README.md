# Rank #1: does an early 4% dip recover better by the close?

The larger historical sample does **not establish that an early 4% dip is a better entry**. The baseline research model's first from-open dip averaged **+0.15%**, while the research model with all 21 dip features averaged **−0.42%**. Their uncertainty intervals include both gains and losses. Early-versus-late differences change direction between the two models and also span zero. This evaluates a particular entry rule; it does not establish that dip measurements are useless as model inputs.

All returns below are **price returns with zero trading expense**: final regular-minute close divided by entry-minute open, minus one. The threshold is fixed at 4%; no favorable threshold, time, ticker exclusion, or exit was selected after seeing outcomes.

The primary assumption is that ranking is available at **8:35 a.m. Chicago / 9:35 a.m. ET**, after the actual opening price is known. Recorded run times remain diagnostics; a later rerun does not establish when that day's first ranking was available. The study uses the current session's official raw open as the primary reference.

## Rank #1 results

One first eligible completed-minute dip per stock-day, with entry at the immediately following minute's open and exit that afternoon:

| Separate cohort | Candidate days | Completed entries | Winners | Mean entry-to-close | Median | Weekly 95% interval for mean |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Actual saved live Rank #1 | 37 | 11 | 8 | +1.57% | +1.42% | −1.76% to +4.69% |
| Historical baseline Rank #1 | 342 | 118 | 64 | +0.15% | +0.50% | −1.44% to +1.74% |
| Historical all-dip-features Rank #1 | 342 | 116 | 58 | −0.42% | +0.11% | −2.30% to +1.23% |

The baseline cohort has 190 no-entry days and 34 unavailable days; the all-dip-features cohort has 197 and 29, respectively. The live cohort has 24 and 2. Unavailable prices are unknown, never counted as cash. The means above condition on a completed entry; they are not returns on every candidate day or compounded portfolio returns.

The two historical models cover the **same 342 dates**, July 1, 2025 through November 6, 2026, and select the same stock on 150 dates. Their combined selections contain 534 distinct stock-days, not 684 independent market dates. The actual live history covers September 18 through November 9, 2026. Across all cohorts, including lower-rank context, there are 869 memberships, 713 distinct stock-days, and 343 distinct dates. Checking a stock at nine times does not create nine new independent examples.

Historical baseline means 132 raw model inputs; all-dip-features means those inputs plus all 21 proposed dip features, or 153. These are the previously frozen research models, **not the current production model or a 19-feature model**. Each was trained strictly before its test block: training ends June 30, 2025; December 31, 2025; and June 30, 2026 for the following three blocks. The study reuses their frozen scores without another fit. It is retrospective research, not actual live history or a fresh prospective holdout for the overall research process. Actual saved live files are mutable, so their original first-published versions are not proven.

## Does an earlier dip help?

“Early” was fixed before outcomes as entry before **9:30 a.m. Chicago / 10:30 a.m. ET**. “Later” begins at that time. These are from-open dips, so overnight gaps are not mistaken for fast intraday declines.

| Cohort | Early entries / mean | Later entries / mean | Early minus later | Weekly 95% interval for difference |
| --- | --- | --- | ---: | --- |
| Actual saved live Rank #1 | 9 / +1.13% | 2 / +3.52% | −2.39 percentage points | Too few later cases |
| Historical baseline Rank #1 | 75 / +0.13% | 43 / +0.19% | −0.06 percentage points | −2.93 to +2.59 points |
| Historical all-dip-features Rank #1 | 81 / −0.30% | 35 / −0.68% | +0.37 percentage points | −2.62 to +3.15 points |

Comparing signals of similar depth does not establish an early premium either. In the predeclared 4–6% depth band, baseline early/later means are +0.05% (60 entries) and +0.17% (42); all-dip-features means are −0.12% (65) and −0.72% (34). Both difference intervals span zero. Deeper-dip late cells contain at most one completed entry, so they cannot support a stable timing comparison.

The first observed threshold crossing is stored separately from the entry time. A decline before ranking availability keeps its original observed onset; it is not relabeled a late decline. The records also distinguish the first dip from a new dip episode after recovery. Known-at-entry fields never contain a later, unobserved onset.

## Fixed-time checks

At each clock, the signal is the immediately preceding completed minute's close. Entry is that clock's actual minute open. The cells below show **completed count / mean return to close**, using the from-open reference. “Not down 4%” includes all other observed price states at the same clock. All-candidate means and joint weekly uncertainty intervals are retained in `results.json` alongside these groups.

| Chicago time (ET) | Baseline down ≥4% | Baseline not down 4% | All-dip-features down ≥4% | All-dip-features not down 4% |
| --- | --- | --- | --- | --- |
| 8:35 a.m. (9:35) | 22 / −1.67% | 252 / +0.74% | 23 / −2.87% | 256 / +0.48% |
| 9:00 a.m. (10:00) | 37 / −0.39% | 230 / +0.40% | 40 / −1.36% | 232 / +0.23% |
| 9:30 a.m. (10:30) | 36 / −0.77% | 227 / −0.11% | 41 / −0.89% | 225 / −0.10% |
| 10:00 a.m. (11:00) | 45 / −0.88% | 216 / −0.11% | 47 / −1.23% | 221 / −0.17% |
| 10:30 a.m. (11:30) | 48 / −0.19% | 206 / −0.26% | 44 / −1.82% | 213 / −0.25% |
| 11:00 a.m. (12:00) | 49 / −0.22% | 208 / −0.32% | 51 / −0.87% | 209 / −0.45% |
| Noon (13:00) | 50 / +0.52% | 204 / +0.08% | 46 / −0.18% | 204 / −0.13% |
| 1:00 p.m. (14:00) | 51 / −0.05% | 198 / +0.18% | 51 / −0.22% | 202 / +0.00% |
| 2:00 p.m. (15:00) | 49 / −0.24% | 206 / +0.19% | 56 / −0.23% | 201 / +0.06% |

For both research cohorts, all nine same-clock dip-minus-nondip weekly intervals include zero. These checks avoid choosing the day's best dip retrospectively, but they remain observational comparisons between different stock-days. A lower entry mechanically improves return if the eventual close is held fixed; it does not guarantee that the lower price forecasts a rebound.

The frozen entry/onset time bins, depth bands, every clock, and both anchors are all retained in the results. We do not select the most favorable row as a discovered trading rule. More observations of the same stock-day improve path detail, not independent sample size.

## Previous-close reference and the original small table

The previous-close anchor is a separate question because it includes overnight gaps. Full-session results are:

| Cohort | Entries | Winners | Mean | Weekly 95% interval | Entries already down 4% at the open |
| --- | ---: | ---: | ---: | --- | ---: |
| Actual saved live Rank #1 | 18 | 10 | +0.96% | −1.39% to +2.73% | 12 |
| Historical baseline Rank #1 | 236 | 122 | +0.63% | −0.57% to +1.74% | 177 |
| Historical all-dip-features Rank #1 | 227 | 115 | +0.03% | −1.09% to +1.08% | 164 |

Split crossings are unavailable under the prespecified raw-price policy. Other corporate actions can still affect nominal prior-close gaps; an opening gap is not automatically an ordinary intraday selloff. This is one reason the from-open result is primary.

For comparison with the earlier table, restrict entries to before **11:00 a.m. Chicago / noon ET** and retain the 8:35 Chicago availability assumption:

| Cohort / reference | Entries | Winners | Mean entry-to-close |
| --- | ---: | ---: | ---: |
| Live Rank #1 / today's open | 10 | 7 | +1.49% |
| Live Rank #1 / previous close | 15 | 8 | +0.93% |
| Historical baseline / today's open | 94 | 52 | +0.16% |
| Historical baseline / previous close | 229 | 119 | +0.64% |
| Historical all-dip-features / today's open | 95 | 49 | −0.36% |
| Historical all-dip-features / previous close | 218 | 111 | +0.01% |

The old 8-entry/+3.01% and 13-entry/+2.40% results used recorded run times, including the special November 9 rerun. Those timestamps do not establish first morning ranking availability. The old sources and protocol remain preserved as a recorded-run timing sensitivity in [the earlier study](../rank_dip_entry_2026_11_09/README.md). The revised live numbers above reproduce an independent calculation on the same frozen prices; no expense was introduced or removed to obtain them.

Availability sensitivities at 8:31 and 8:45 Chicago (9:31 and 9:45 ET) are also retained. They are alternate assumptions, not claims about actual publication. The research from-open mean is +0.14%/+0.15%/+0.39% for baseline and −0.41%/−0.42%/−0.37% for all-dip-features at 8:31/8:35/8:45, respectively. No best-availability assumption is selected.

## Coverage, downside, and interpretation

There are 713 minute-source sessions, including one empty response, and 1,388 daily-reference keys, of which six remain unavailable. All raw source hashes and provider identities were audited. Of the 713 distinct stock-days, 317 have every expected regular minute and 396 have at least one absent minute. Absence can mean no qualifying trade; it is not filled with an invented price. Exact signal/entry and expected final exit bars are required. Session boundaries use the frozen exchange calendar, including half-days and DST.

The source-opening audit compares official raw daily open with the first regular-minute open before evaluating outcomes. All 673 comparable stock-days match exactly; no case was excluded by the predeclared greater-than-1% discrepancy rule. MOG-B uses a dated-verified provider alias, MOG.B, without rewriting the raw payload. Its missing prior daily reference remains unknown. Split metadata is complete for the study period, including that alias; the previous-close crossing test is `previous_session < split_date <= current_session`.

For primary whole-session from-open results, unavailable reasons are: baseline 25 missing opening minutes, 7 missing exact entry minutes, and 2 missing final exit minutes; all-dip-features 22, 5, and 2; live 2 missing opening minutes. Before-noon denominators differ where the first observed qualifying signal occurs later in the day. The complete reason counts for every scenario remain in `results.json`.

The worst historical entry in both models is QH on July 28, 2025: entry $0.4103, final-minute close $0.122, or **−70.27%**. Its 390 regular bars are present; official daily open $0.559 matches the first minute. The provider's official daily close is $0.118. Separately cached Yahoo daily prices imply the same −78.89% open-to-close change despite their adjusted price scale, independently corroborating that day's collapse. This does not independently verify the exact minute-entry return or broker fills. The daily close and final-minute proxy differ, so they must not be described as identical executions. No split crosses that prior-close reference. This case remains in the sample; no new price adjustment is applied. QH's separately flagged April 2026 price-basis issue is not resolved by this July comparison. The median is reported alongside the mean because tail losses matter.

Mean observed further drawdown after entry is −4.80% for baseline, −5.51% for all-dip-features, and −4.28% for live Rank #1. The worst observed further drawdown is −74.02% for both historical models and −13.42% for live. These are outcomes from subsequent observed minute lows, not features known at entry or guaranteed executable stop prices. No leverage or position sizing is modeled.

The 148 actual saved ranks 2–5 are retained only as secondary context and do not drive the Rank #1 conclusion. MRCY is included on November 4 at rank 2; no ticker was hand-excluded. This study cannot establish the extra value of being Rank #1 relative to an otherwise matched lower-ranked dip.

Weekly intervals resample whole ISO calendar weeks, preserving within-week dependence and repeated clocks. There are 60 completed-entry weeks for baseline's primary from-open result, 62 for all-dip-features, and 7 for live. Each interval uses 2,000 replicas and fixed seed 20261109. An interval is suppressed if a group has fewer than eight entries or five completed weeks. These are descriptive, pointwise intervals across many prespecified views, not a multiple-comparison-adjusted discovery claim.

The useful output for feature iteration is a reproducible record of **dip depth, time since the first dip, current dip-episode age, whether a recovery already occurred, and minutes remaining**. Those are available at a decision time and can be evaluated as ordinary inputs to a future remaining-session model. The present evidence does not justify hard-coding “Rank #1 plus a 4% fall means buy,” or “earlier always means better.” It also does not evaluate the current production feature subset. No model, UI, trading rule, database, or service was changed by this study.

## Reproduce and inspect

The protocol was frozen at 03:56:24 UTC on November 10, 2026; the explicit opening-source check was added at 03:59:48 UTC, before any expanded outcomes. Cohort and acquisition provenance retain the research checkpoint hashes, training dates, raw-price request contracts, terminal statuses, and source hashes. Original archived request pages were not retained for some reused minute sources; this limitation is labeled in acquisition metadata.

Offline analysis and tests require only Python's standard library and the frozen repository sources:

```sh
python3 research/rank_dip_timing_2026_11_09/analyze.py
python3 research/rank_dip_timing_2026_11_09/test_analysis.py
```

`results.json` contains all 48 first-event summaries, 128 timing/depth summaries, 216 checkpoint summaries, and 104 comparisons. `event_cases.json.gz` retains 10,428 event-scenario records; `checkpoint_cases.json.gz` retains 15,642 clock-scenario records. These 26,070 records are scenario observations, not 26,070 independent trades. Their manifests contain byte hashes; gzip output is deterministic. Regenerating `results.json` changes its generation timestamp but preserves numerical results for unchanged inputs.

Twenty focused tests cover opening-price availability, completed-bar timing, first onset versus entry, opening gaps, exact next-minute execution with no retry, missing boundaries, half-days/DST, future-onset leakage, repeated episodes, source-price consistency, verified aliases, split-date boundaries, weekly clustering, and zero-expense arithmetic. The independent review reconciles every raw case and summary, using a separate implementation. `audit_checks.json` records count, as-of timing, return-arithmetic, and artifact-hash checks.
