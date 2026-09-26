# 0013: Every training run's model is archived; the live model is a user pick

Date: 2026-09-08 (backfilled 2026-09-26)
Status: Accepted

## Context

Retraining used to overwrite the only pickle. A bad run (or a bad feature
change) silently became production, with no way back and no comparison.

## Decision

Every run's ensemble is archived with its metrics in TrainingRunStore
(version-controlled, #50); the Models tab picks which archived run is live
(#78, hardened in #123). Nightly retrains append, never destroy.

## Consequences

- Rollback is a click, and "did this change help" is a table, not a memory.
- MLflow tracking (SQLite) rides along for run metrics; it is a mirror, not
  the source of truth.
