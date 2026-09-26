# 0017: Morning quotes: Polygon, then Yahoo, then Finnhub; blacklist store

Date: 2026-09-23 (backfilled 2026-09-26)
Status: Accepted

## Context

Scoring ~2000 names in the minute after the open needs opens for all of
them; no single free/entitled provider reliably covers that at speed.

## Decision

Modular provider chain (#118): Polygon full-market snapshot when entitled,
Yahoo for the rest, Finnhub for leftovers; fetching and scoring run in
200-name parallel buckets (#102). A TickerBlacklistStore (#117) removes
names that structurally break quoting.

## Consequences

- Provider swap/reorder is config-shaped, not a rewrite.
- Quote freshness/plausibility checks stay in inference (0003), not in the
  providers.
