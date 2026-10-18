# 0025: Record shadow Jev confidences on every morning pick

Date: 2026-10-18
Status: Implemented (shadow only)

## Decision

The live 8:30 run asks TypeSafe Jev three yes/no questions about each Rank
and Fit pick, after news and before the final publish:

- `news_skip`: material news, so avoid
- `buy_today`: closes above today's open
- `hold_overnight`: tomorrow's open is above today's close

`training/jev_shadow.py` sends one SystemOne call per pick with only
open-known facts: list and rank, model score, gap, eight recent feature
values, and the same headlines Grok sees. The answers are saved as `jev` on
each pick in the scan and shown in a Jev column. **Nothing reads them back**
into picks, skips, sizing or What If. Replays, test runs and quote recovery
pass no scorer, so they spend no tokens. Failures are recorded as an `issue`
on the pick and never break the scan.

Jev is on the same LLM proxy host and key as the Grok judge, at
`/typesafe/v1/systemone` (not under the proxy's `/v1` base).

## Evidence so far

All 305 saved news decisions (Sep 21 to Oct 16) ran through Jev in 27s,
with no errors and about 900 tokens per call. Results:

| Group | n | Mean day return | Win rate |
|---|---|---|---|
| Current judge AVOID | 58 | +0.71% | 53% |
| Current judge OK | 247 | -0.62% | 46% |
| Jev avoid >= 0.5 | 108 | -0.16% | 46% |
| Flagged only by Jev | 54 | -1.04% | 41% |

- Jev agreed with the current judge on 247 of 305 picks, and flagged 54 more.
- The names flagged only by Jev did worse on the day (-1.04%, 41% wins) than
  the names neither judge flagged.
- AVOID names from either judge moved more (mean |return| 3.3-3.6% against
  2.6-2.7%) but were not losers on average, so the current skip may be giving
  up winners.

This is one month of data. It is not evidence for wiring Jev into skips.

## Before acting on it

Collect 2-3 weeks of forward scores. Then score each question against what
actually happened: the open-to-close return for `buy_today`, the next open
against the close for `hold_overnight`, and absolute move for `news_skip`.
Promote a question only if it beats the model and judge it would replace.
Record that decision in a new ADR.

Costs are unknown: TypeSafe publishes no pricing, and only the test proxy
is verified. At about 25 picks and 900 tokens each, the morning run uses
about 23K tokens a day.
