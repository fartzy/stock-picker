# 0015: Rank is its own top-K pickle, never blended with Fit

Date: 2026-09-16 (backfilled 2026-09-26)
Status: Accepted (objective superseded by 0018)

## Context

Fit (regression, 0.5% gate) answers "is this worth betting on at all";
a ranking model answers "which 20 look best relative to each other today."
Rank scores are relative, not percents -- averaging them with regression
outputs is a type error.

## Decision

Train a separate ranking model (#90: lambdarank) persisted as its own
pickle (`day_session_return_rank.pkl`), served as its own morning list
(top-20, news checked on the top 10), never entering the return ensemble.
`partition_model_specs` enforces the separation at the spec level.

## Consequences

- Two lists every morning with different semantics, which later enables
  consensus experiments between them (0019).
- The rank objective itself is swappable behind the same pickle name --
  exactly how 0018 landed without touching serving.
