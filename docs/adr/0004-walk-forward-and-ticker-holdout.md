# 0004: Walk-forward validation plus held-out tickers; never k-fold

Date: 2026-09-03 (backfilled 2026-09-26)
Status: Accepted

## Context

k-fold shuffles rows, which for pooled time-series data trains on the future
to predict the past -- backtests look great and mean nothing.

## Decision

Two independent generalization checks:
- Date-based walk-forward folds (`training/splits.py`): unique dates are
  partitioned into contiguous blocks; every train date strictly precedes
  every test date.
- ~10% of tickers (deterministic seed) held out of training entirely and
  evaluated once, at the end -- never inside a search loop.

## Consequences

- All tuning/search scripts decide on walk-forward folds only; the holdout
  answers "does the winner generalize to unseen stocks," and stays honest
  only if searches never select on it.
- Holdout scoring shares calendar dates with training (the split is by
  ticker), so holdout hit rates run higher than live; compare candidates
  relative to each other, not to live numbers.
