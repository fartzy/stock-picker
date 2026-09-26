# 0019: Proposed: consensus gate (Fit 0.5% ∩ Rank top-20) as a stricter buy list

Date: 2026-09-26
Status: Proposed (tooling landed in PR #129; no production gating change yet)

## Context

Fit and Rank optimize structurally different losses (per-row magnitude vs
within-day order). Live Sep 8 validation showed hit rate rising steeply
with confidence. Requiring both lenses to agree is a stricter, nearly free
confidence gate.

## Evidence so far

`backtest.simulate_consensus` (walk-forward + holdout, ListFold as Rank):
holdout mae∩listfold = 729 trades, 83.0% hit, +2.34% avg vs the plain 0.5%
gate's 2,926 trades, 80.8% hit, +1.84% avg. Fold-level: 59.4% vs 52.1% hit.

## Next step before accepting

Surface agreement in the morning UI (e.g. mark Fit names that are also in
Rank's top-20) and paper-track it live for a few weeks via the What if book
before changing any default gating.
