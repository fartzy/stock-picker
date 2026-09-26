# 0002: Five layers, each talking only to the one below

Date: 2026-09-02 (backfilled 2026-09-26)
Status: Accepted

## Context

Data flows one way: symbols are ingested, prices stored, features computed,
models trained, predictions served. Letting layers reach around each other
(e.g. API code fetching prices) makes both testing and reasoning harder.

## Decision

`ingestion/ -> storage/ -> features/ -> training/ -> api/ -> typescript/`,
each layer depending only on the one below it. API routes wrap
already-tested pure functions and hold no business logic of their own.

## Consequences

- Any layer is testable with fakes for the layer below (see 0005's data_dir
  injection).
- New capability lands in the lowest layer that can own it, then gets a thin
  route + UI on top -- most PRs in the history follow that shape.
