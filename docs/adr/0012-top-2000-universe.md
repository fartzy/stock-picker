# 0012: Universe = top ~2000 US names by market cap

Date: 2026-09-08 (backfilled 2026-09-26)
Status: Accepted

## Context

The original 500-name universe limited both training rows and the
cross-section a ranking model can exploit. Expansion (#68-#73) hit real
constraints: Yahoo rate limits, symbol-directory parsing edge cases ("NA").

## Decision

Top ~2000 by market cap from NASDAQ+NYSE symbol directories, with a manual
additions list and (later, #117) a blacklist store for structurally broken
tickers. Feature pruning was re-derived for the larger universe (#81).

## Consequences

- ~443k training rows as of late Sep 2026; per-day cross-sections of ~1800
  names, which is what makes listwise ranking objectives viable (see 0018).
- Ingestion cost and provider rate limits become the binding constraint on
  further growth.
