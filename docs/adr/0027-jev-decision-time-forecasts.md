# 0027: Preserve Jev evidence and assess explicit decision-time targets

Date: 2026-11-09

Status: Accepted; forward evidence collection, with no automatic trade gate.

## Problem

The original Jev `buy_today` probability answers whether the stock closes above
its **session open**. It does not answer whether entering at a later price earns
a positive return. A Rank score is also a relative ordering score, not an
expected return or a calibrated probability.

Morning quote recovery constructed fresh signals without their Jev answers,
then replaced the day's scan. That lost forecasts needed for forward evaluation.
Recreating answers after their outcomes are known would not repair the evidence.

## Decision

Keep the original three morning questions and field meanings from ADR 0025.
Archive each new request independently in `data/jev/forecasts.db` before its
answer is attached to a mutable scan. Records contain the exact question set,
input state, request/completion and evidence times, model, probabilities,
coverage, original list rank and score, and target definition. An existing ID
cannot be replaced or deleted through the store; identical retries are
idempotent. Later assessments receive new IDs. Quote recovery preserves original
embedded answers and only asks for newly appearing candidates.

Add two manual assessment modes, separate from the morning probabilities:

* `entry_v1`: positive net return from the stated entry to today's regular close,
  after an explicit round-trip cost per share; and whether price reaches the
  stated downside barrier **before** that close. The barrier is
  `entry_price * (1 - downside_fraction)`, a gross price level rather than a
  terminal loss test. The interval begins when the answer is available.
* `overnight_v1`: whether the next XNYS session opens above today's actual
  regular-session close plus the explicit cost. Ask only in the final 30 minutes
  before the exchange's actual close, including early closes. The current trade
  is contextual evidence; it is never substituted for the eventual close.

Entry prices are labeled either a dated observed trade or a user assumption.
An observed trade is not an executable ask or a guaranteed fill. Zero entered
cost means a zero-cost assumption, not a claim that execution is free. No
assessment places orders, changes Rank/Fit membership, skips picks, or sizes
positions.

## Evidence constraints

Use the XNYS calendar for holidays, early closes and the next session. Require a
regular-session observed trade no more than five minutes old, with timestamps no
later than the evidence cutoff. An explicit user entry assumption may proceed
without a valid observed trade, with the missing quote recorded.

Capture the evidence cutoff after fetching quotes and news, then exclude future
or undated articles. Daily features must precede the assessment's session;
stale or unavailable inputs remain explicit. Save news source coverage, issues,
check time and normalized articles. A degraded or missing feed is never rendered
as a successful no-news check. Keep the original morning news judgment separately
for later comparisons.

Reject malformed, nonfinite or out-of-range SystemOne probabilities. Preserve
sanitized failure codes and completed timestamps. A response arriving after the
session close remains in history but is ineligible as a live forward assessment.

## Evaluation

Evaluate all archived candidates, including untraded names. Never ask Jev again
to label past records. Refresh outcomes using dated provider prices; retain
pending or unavailable outcomes explicitly. Downside requires post-assessment
minute-path evidence, not the full day's low. A partial starting minute cannot
prove a post-assessment touch; incomplete coverage cannot prove no touch.

Keep morning, entry and overnight targets, model versions, list kinds, price
bases, costs and barrier settings in distinct cohorts. Morning forecasts made
after the open may be assessed for their stated close/open event, but their
open-to-close return is hypothetical and must not be presented as an available
entry trade. Older answers without original input snapshots are legacy evidence,
not reconstructed forward forecasts. Repeated manual requests remain visible;
the first valid eligible record defines the primary observation.

Report sample size, Brier score, probability-bin reliability and realized
conditional returns alongside prior-day-only base rates and Rank/news
comparisons. Sparse bins and same-day outcomes do not establish calibration or
justify a new trading threshold. Automatic promotion still requires chronological
evidence under the existing research rules.

## Verification

Tests cover immutable/concurrent writes, preservation through recovery using new
signal objects, original morning question semantics, valid probability bounds,
current-entry and actual-close overnight targets, explicit costs, future/stale
quotes, holiday and early-close timing, future news/features, delayed responses,
thin API wiring and outcome evaluation. Run `bazelisk test //...` before merging.
