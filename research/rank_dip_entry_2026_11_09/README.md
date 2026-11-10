# Fresh entry after a 4% drop in a saved Rank #1 pick

The small sample has positive average returns after the dip, but it does **not establish a predictive buy rule**. The eight morning entries with recorded publication times averaged **+3.01% from entry to close**, with six winners. The descriptive 95% bootstrap interval for their mean runs from **−1.24% to +7.22%**. Final saved rankings are mutable, so these are not proven point-in-time selections.

This is a new-entry study. It does not start with an open position, average down, or hold overnight. It also does not retrain or change a model.

## Results with recorded publication times

There are 37 saved Rank #1 sessions from September 18 through November 9, 2026. Publication times are available for 28; nine remain unavailable in this sensitivity. Each signal uses the first qualifying **whole minute after publication**, then the next minute's open. The price may already be more than 4% below the reference when the ranking becomes available. Entries must occur strictly before noon ET.

| Reference for the 4% drop | Completed entries | Winners before costs | Mean before costs | Median before costs | Mean after 0.25% round-trip costs | Worst before costs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Today's raw session open | 8 | 6/8 (75%) | +3.01% | +2.52% | +2.76% | −8.15% |
| Previous regular-session raw close | 13 | 10/13 (77%) | +2.40% | +1.80% | +2.15% | −8.15% |

The previous-close mean's descriptive bootstrap interval is **−0.25% to +5.21%**. After the 0.25% cost assumption, its winners fall to 9/13; the from-open winners remain 6/8. These intervals span zero, use very few entries, and do not account for selection-version uncertainty or dependence between sessions.

There were 20 no-entry sessions for the from-open rule and 15 for the previous-close rule among the 28 dated opportunities. Treating those as cash, while keeping the nine unavailable sessions out of the denominator, gives after-cost means of +0.79% and +1.00% per known opportunity. These are arithmetic averages, not compounded portfolio returns or a comparison against buying every morning.

## Sensitivity to assuming the ranking was known at the open

This separate calculation uses all 37 final saved selections with a hypothetical availability time of 9:30 a.m. ET. It must not be described as a feasible historical strategy. Two sessions lack the required opening minute: PS on September 21 and TDTH on September 25. They are reported as unavailable, not counted as cash or dropped by ticker preference.

| Reference for the 4% drop | Completed entries | Winners before costs | Mean before costs | Median before costs | Mean after 0.25% costs | No-entry days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Today's raw session open | 10 | 7/10 | +1.89% | +1.15% | +1.64% | 25 |
| Previous regular-session raw close | 15 | 7/15 | +0.43% | −0.91% | +0.18% | 20 |

The large change when entry timing changes is a reason to keep the publication limitation prominent. The two scenarios use different timing and available samples; their difference is not an estimate of the benefit of waiting for publication. Whole-session sensitivities and 0/10/25-bps cost results are retained in `results.json`, without choosing a winning variant.

## What the calculation does and does not show

At an unchanged eventual closing price, a lower entry mechanically produces a higher return: `close / entry - 1`. That does not prove the decline predicts a recovery. The relevant forecast would be the remaining return conditional on the observed decline, time, news, and other information. A morning Rank #1 score is an ordering, not a calibrated remaining-day expected return.

The current evidence justifies investigating that conditional-entry question further. It does not show that Rank #1 adds value after a decline compared with other equally declined stocks, nor that this timing beats a suitable predeclared alternative on unseen data. No automatic buy/skip rule was added.

No ticker was hand-excluded. MRCY was not Rank #1 on any of these 37 saved sessions, so this particular cohort does not evaluate the MRCY example. The cohort definition was fixed to the user's Rank #1 question before outcomes.

## Frozen execution and data rules

- `protocol.json` was written at 03:09:41 UTC on November 10, before outcomes. The threshold is exactly 4%; there is no parameter search.
- A trigger requires a completed minute **close** at or below 96% of the reference. A low touching the threshold is insufficient. Entry uses the immediately following minute's actual open, even if that open rebounds above the threshold. Missing that minute makes the entry unavailable, with no later retry.
- The expected final regular-session minute's close is the exit proxy. It is not a guaranteed closing-auction or broker fill. `signal_at` and `exit_at` are bar-start timestamps; `signal_observed_at` and `exit_close_observed_at` identify when their closing prices become observable.
- Missing session boundaries, invalid/nonfinite OHLC, duplicates, and unavailable entry/exit bars are explicitly recorded. No price is forward-filled. Sparse intermediate bars are retained; a missing provider aggregate can mean no qualifying trades, not necessarily a feed outage. “No entry” means no qualifying **observed completed bar** under this rule.
- Costs subtract a fixed 0/0.10/0.25 percentage point round trip from each entry return. There is no additional inferred fill improvement, borrow, spread, latency, market impact, or liquidity guarantee. No leverage or position-sizing model is assumed.
- All original selections, raw scores, source hashes, session bounds, and available publication times are frozen in `cohort.json`. The original ignored daily Rank JSON files can be overwritten; this study cannot reconstruct their first published versions. November 9's logged publication is 10:07:27 ET, and its first eligible signal bar starts at 10:08, with entry at 10:09.

## Source and verification limits

The study reuses 36 full-OHLC unadjusted minute archives from `research/averaging_down_2026_11_06/rank_reconciliation/minute_bars/` and fetches the single missing IONS November 9 session through the existing Massive client. The production `MinuteBarCache` is not used because it drops open/close prices. Existing fetch manifests confirm successful archive requests, and the client implements bounded pagination; the archived documents do not retain original response pages/status fields.

Raw daily anchors use independently fetched Massive regular-session open/close summaries, including 35 bounded daily requests; payload status, ticker, and date must match. Previous-session dates and cash-session bounds use the existing XNYS calendar helper, including DST. Saved `prev_close` values are retained for provenance but are not used as the study's reference prices.

Before outcomes, `protocol_clarification.json` added one bounded splits-metadata request. The returned page contained 259 dated events, no cursor, and no event for a cohort ticker in the checked period. The retained source has a response hash, filters, count, and relevant-event list; the original payload status was not retained, so this run's split-status provenance is limited. Future fetches now require and retain `status == "OK"`. Split crossings are marked unavailable rather than adjusted with an invented ratio. This check does not certify absence of dividends, spinoffs, ticker reuse, or other corporate actions; the previous-close variant remains a nominal raw-price comparison.

Provider contracts: [minute aggregates](https://massive.com/docs/rest/stocks/aggregates/custom-bars), [stock splits](https://massive.com/docs/rest/stocks/corporate-actions/splits). Prices are qualifying-trade aggregates, not execution quotes.

Nine synthetic checks cover completed-close triggering, next-minute execution including a rebound, no later retry after a missing entry, the noon cutoff, publication-minute boundaries, unavailable exits, regular-session/DST bounds, and rejection of malformed early bars. Independent reviewers checked the results and source artifacts separately.

## Reproduce

Offline analysis uses only this folder's frozen cohort/daily sources and the existing tracked minute archives. It does not reread mutable daily Rank selections or request prices:

```sh
task_python=bazel-bin/python/stock_picker/api/main.runfiles/rules_python++python+python_3_12_aarch64-apple-darwin/bin/python3
"$task_python" research/rank_dip_entry_2026_11_09/study.py analyze
python3 research/rank_dip_entry_2026_11_09/test_study.py
```

The analyzer uses the already-built API's pinned dependency directories; no package is installed. `study.py fetch` is a separate, explicitly network-enabled refresh step and rebuilds the cohort from mutable ignored Rank JSON files, so it is **not** the command for reproducing this frozen study. `study.py splits` uses the cached metadata when present. A future fresh data collection should use a new study folder/protocol rather than overwrite this evidence.

`results.json` contains all eight predeclared summaries and 296 per-session scenario records. `audit_checks.json` records hashes, count reconciliation, and verification evidence.
