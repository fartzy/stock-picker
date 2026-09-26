# 0016: News judged locally (Langfuse-traced); skip rules are conditional on the gap

Date: 2026-09-20 (backfilled 2026-09-26)
Status: Accepted

## Context

A cheap pre-open check: does overnight news make a pick untradeable? Early
versions skipped too eagerly -- bad-news names that gapped UP at the open
were being dropped even though the market had already shrugged.

## Decision

Finnhub news is ingested for the whole universe nightly (news.db); a
material-news judge runs on the short list each morning, traced through a
local Langfuse (#94). Skip rules are gap-conditional: news names are skipped
only when the open gaps down (#104); insider sells are skipped even on a
gap-up (#105).

## Consequences

- The judge only sees the short list (top names), keeping the 8:30 window
  fast.
- Skip reasons are recorded on the scan so the UI can say why a name is
  missing (0011's "empty list is the design working" applies here too).
