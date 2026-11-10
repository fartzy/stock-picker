# 0028: Separate current-entry-to-close helper

Date: 2026-11-10

Status: Accepted

## Context

The morning Rank/Fit models describe entry at the opening price. A later drop
changes both entry price and the information available about the day's path.
Applying the original score to a later entry does not estimate its conditional
odds. The user wants fast reassessment of 10–20 saved morning candidates, with
this model managed separately from the crowded morning-model controls.

## Decision

Create a separate `intraday_entry` artifact and Models → Intraday Entry panel.
Its outcomes are probability of a positive observed-entry-to-cash-close return,
expected remaining return, and probability of losing more than 2% at that close.
All targets use zero assumed expenses. The downside label is a terminal outcome,
not an intraday barrier-touch probability, and none of the outputs places orders.

Training consumes frozen, provenance-checked research ranks plus already cached
raw minute prices. It includes drops, flat paths, and rises. Rank 1 is the primary
evaluation; cached lower-rank coverage is sparse and selected. Scores are scaled
within each morning's top 20. This reduces numerical scale changes but does not
remove the domain shift from the historical research Rank model to current Rank.

A shared pure feature builder accepts only completed bars as of entry and a
strictly prior-session feature snapshot. Missing coverage remains unknown. Fixed
historical checkpoints enter at the next minute's open; the target uses that
session's final-minute close. Current predictions use a timestamped recent trade,
the verified regular-session open, and only bars completed by that trade time.

Whole sessions split chronologically into 60% estimator fit, 20% sigmoid
calibration, and 20% final evaluation. Repeated snapshots share each ticker-day's
weight and ticker-days share each date's weight. Missing/constant feature pruning
uses fitting rows only. Three small LightGBM heads learn profit, mean return
(L2 objective), and severe loss conditional on a nonpositive return. Multiplying
the last head by one minus profit probability keeps disjoint outcomes coherent.
One-class subsets fall back to their fitting empirical rates and report the
limitation. A training-only logistic model and per-clock empirical rates are
benchmarks. Reliability bins, Brier/log loss, return error, class counts, Rank 1
results and lower-rank coverage remain visible.

The exact evaluated artifact is persisted without refitting on evaluation rows.
Fit-through, calibration-through, and evaluated-through dates are distinct.
This is experimental evidence rather than a claim of a profitable strategy.

Live refresh has a 20-name limit, six fetching workers, a short market-data cache,
and an artifact cache invalidated by atomic ModelStore publication. Training is a
separate bounded background job. No morning model or saved scan is overwritten.
Closed sessions, stale quotes, missing opening minutes, missing saved ranks, and
provider failures return explicit unavailable rows rather than fabricated odds.
The saved 09:35–15:00 ET checkpoint range and minimum 60-minute remaining
horizon also bound live use, including an earlier cutoff on early-close days.

## Consequences

The helper can refresh a small list without retraining the universe. Its artifact
has a real chronological holdout and can be inspected independently. Evaluation
checkpoints are correlated observations, so stock-day/session counts accompany
row counts. Research-to-production rank drift, sparse lower-rank coverage, raw
last-trade entry proxies, and limited recent history remain limitations. Future
model versions must collect forward evidence or reserve another untouched block;
iterating repeatedly on this reported holdout would no longer be independent.

Library references: [LightGBM objectives](https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst),
[scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
