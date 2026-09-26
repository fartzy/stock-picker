# 0003: Same-day open→close label; shift close-known features, never open-known

Date: 2026-09-02 (backfilled 2026-09-26)
Status: Accepted

## Context

The product question is "what should I buy at the open and sell at the
close?" Most feature columns are computed from a day's full OHLCV -- using
day t's own features to predict day t's close-derived label would leak the
answer into the question.

## Decision

Label = (Close_t - Open_t) / Open_t. Features are shifted one day
(features through t-1 predict day t), EXCEPT the open-known columns
(overnight gap, recency/seasonality patterns keyed off completed days plus
Open_t), which are legitimately knowable at the open and stay unshifted.
Live scoring rebuilds exactly that row shape: last night's persisted
snapshot plus today's live open (`training/inference.py`), raising on
stale/implausible inputs instead of silently scoring.

## Consequences

- `training/dataset.py` is the single place this contract lives; read it
  before touching anything else in training (per CLAUDE.md).
- Every new feature must be classified close-known vs open-known; a
  mistake there is a leak, which is why the sets are explicit constants
  (`OPEN_KNOWN_FEATURE_COLUMNS`).
