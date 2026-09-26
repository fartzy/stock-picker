# 0011: Fit trades only above a 0.5% predicted-return confidence gate

Date: 2026-09-07 (backfilled 2026-09-26)
Status: Accepted

## Context

Scoring ~2000 names every day always produces a ranking, but most days most
predictions are noise-sized. Live Sep 8 validation confirmed the mechanism:
hit rate at a 1.0% cut ran far above the ungated list.

## Decision

The Fit list only shows names whose predicted open→close return exceeds
0.5%; sweeps report n_trades next to every metric so selectivity and sample
size stay visible together. A volatility-normalized (z-score) gate was
prototyped (#54) and kept as a diagnostic, not promoted -- the fixed
threshold stayed simpler and the z-gate's stability advantage wasn't
demonstrated.

## Consequences

- Some days the Fit list is empty. That is the design working.
- Threshold-vs-regime drift remains a known open question (the z-gate
  diagnostic in tune_experiment.py exists to revisit it).
