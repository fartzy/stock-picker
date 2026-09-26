# 0009: Every trainer is seeded; run-to-run variance is a bug, not noise

Date: 2026-09-07 (backfilled 2026-09-26)
Status: Accepted

## Context

LightGBM's feature/bagging subsampling drew fresh randomness every run
(#59): two trainings on identical data produced different holdout numbers,
which read as real improvements or regressions during tuning.

## Decision

Fixed seeds across all trainers (LightGBM `seed: 0`, sklearn
`random_state: 0`), plus deterministic holdout-ticker selection.

## Consequences

- A metric change between runs now means the data or config changed.
- Reproducibility is asserted in tests
  (`test_train_lightgbm_is_reproducible_across_runs`).
