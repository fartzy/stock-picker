# 0007: Every feature ships prose, formula, and worked example, completeness-tested

Date: 2026-09-04 (backfilled 2026-09-26)
Status: Accepted

## Context

~100 engineered columns are unauditable if their meaning lives only in
pipeline code. The owner wants to check any feature's math by hand.

## Decision

Three parallel, pattern-matched, column-name-driven views must stay in sync
for every real feature: `descriptions.py` (plain English), `formulas.py`
(the actual pandas expression), `examples.py` (a worked numeric example).
Each has its own completeness test that fails if any real column falls
through to an "unknown" placeholder.

## Consequences

- Adding a feature means adding it three more places -- deliberate friction
  that keeps the catalog trustworthy.
- The tests make "I forgot to document it" a build failure, not a review nit.
