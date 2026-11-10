# Paired audit: a lower entry does improve the same-stock return

**Yes: buying the same stock lower, then selling at the same price, improves the return.** The earlier headline answered whether the dip entries were profitable and whether early dips beat later dips. It did not show the matched improvement over buying those same stocks at the open. This addendum supplies that comparison without changing the original results.

For the identical closing price `C`, opening price `O`, and lower dip entry `D`, the improvement is `C / D − C / O = C × (1 / D − 1 / O) > 0`. A loss can become smaller without becoming a profit. All numbers below use **zero trading expense**.

| Rank #1 cohort | Same dip days | Buy at open → same exit | Dip entry → same exit | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Actual saved live Rank #1 | 11 | -3.287% | +1.569% | +4.856 pp |
| Historical baseline Rank #1 | 118 | -4.892% | +0.153% | +5.045 pp |
| Historical all-21-dip-features Rank #1 | 116 | -5.544% | -0.415% | +5.128 pp |

Every completed dip entry was below its own session open, so every one improved that same-stock, same-exit return. The signal is a completed-minute close down at least 4%; execution uses the next minute's open. That open rebounded above the exact 4% threshold in 1 live, 12 baseline, and 9 all-21 cases. The observed execution prices are retained, not replaced with an assumed threshold fill.

XENE on 2026-09-18: open $44.02, dip entry $42.07, identical exit $39.74. Return improves from -9.723% to -5.538%, or +4.184 pp.

IONS on 2026-11-09: open $32.27, dip entry $30.80, identical exit $31.03. Return improves from -3.843% to +0.747%, or +4.589 pp.

The opening-price comparison conditions on days that later dipped. It proves the entry-price arithmetic for those days; it does not include days where waiting never triggered. The session open is also a reference price, not an assumed executable ranked trade before ranking availability.

A feasible reference under the frozen assumption is buying at **8:35 a.m. Chicago / 9:35 a.m. ET**. On dip days where both that entry and the dip entry are observed:

| Rank #1 cohort | Matched dip days | Buy at 8:35 Chicago | Wait for dip | Difference |
| --- | ---: | ---: | ---: | ---: |
| Actual saved live Rank #1 | 11 | -0.189% | +1.569% | +1.758 pp |
| Historical baseline Rank #1 | 115 | -2.866% | +0.286% | +3.152 pp |
| Historical all-21-dip-features Rank #1 | 108 | -3.464% | -0.271% | +3.194 pp |

A dip can already be signaled by 8:35, in which case both rules enter at the same price. The 8:35 price may already be below the session open, so this comparison differs from the opening-price comparison.

For the full waiting policy, both alternatives must use the **same candidate days**. Waiting earns zero on an observed no-dip day; an unavailable outcome excludes the day from both alternatives. The unconditional 8:35 buy requires that minute's open and the final exit bar, not an 8:34 closing signal.

| Rank #1 cohort | Matched days (dip / no entry) | Excluded unknown | Buy at 8:35 Chicago: mean/day | Wait: mean/day | Wait minus buy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual saved live Rank #1 | 32 (11 / 21) | 5 | +1.069% | +0.539% | -0.529 pp |
| Historical baseline Rank #1 | 274 (115 / 159) | 68 | +0.457% | +0.120% | -0.337 pp |
| Historical all-21-dip-features Rank #1 | 284 (108 / 176) | 58 | +0.182% | -0.103% | -0.285 pp |

In these matched samples, waiting improved the return on the dip days on average, but its average across all matched candidate days was lower because it also skipped no-dip days. These are arithmetic, equal-capital daily averages, with zero earned on observed waiting days; they are not compounded portfolio returns or proof of a future edge.

The baseline and all-21 research cohorts cover the same 342 dates and stay separate; the live cohort has 37 dates. These are the frozen historical research models, not the current production feature subset. Availability at 8:35 Chicago is an assumption after the opening price is known, not proof of every live day's first publication time. Entry and exit prices are minute-bar proxies, not guaranteed broker fills. This paired comparison answers a different question from whether earlier dips outperform later dips; no new timing rule or parameter search was introduced.

`paired_entry_audit.json` retains every matched pair, unknown exclusion, overlapping exclusion reason, original artifact hash, and raw source hash. Assertions independently reconcile the saved first trigger and next-minute entry with raw bars, identical exits, lower-entry arithmetic, counts, and unchanged original artifacts. The script uses only the standard library and frozen files:

```sh
python3 research/rank_dip_timing_2026_11_09/paired_entry_audit.py
```
