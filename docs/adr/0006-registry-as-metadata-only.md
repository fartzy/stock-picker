# 0006: Feast-style registry is metadata over the pipeline, not a second backend

Date: 2026-09-04 (backfilled 2026-09-26)
Status: Accepted

## Context

A real feature-store deployment (Feast et al.) would add serving
infrastructure this single-machine app doesn't need, but the
Entity/FeatureView/FeatureService vocabulary is useful for organizing ~100
columns.

## Decision

`features/registry.py` layers Feast-style metadata over the same pandas
pipeline that actually computes features. It describes; it never computes or
serves.

## Consequences

- The Feature Store tab (catalog, coverage, prune) reads the registry;
  training reads the pipeline. They cannot drift apart because there is only
  one computation path.
