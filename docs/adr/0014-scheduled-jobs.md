# 0014: Nightly 3:30 PM CT retrain; morning scoring 8:30:15 CT, click takes priority

Date: 2026-09-11 (backfilled 2026-09-26)
Status: Accepted

## Context

The pipeline (prices -> features -> retrain) must complete after the close,
and scoring must happen right after the open with today's real open prices.
launchd on a Mac that must be awake; timezone bugs already bit once (#88).

## Decision

Weekday launchd jobs, Chicago time: 3:30 PM prices/features/retrain (plus
news ingest), 8:30:15 AM backup scoring (Polygon day.o) that a manual click
on Trading supersedes and disarms (#93, #103, #119). Clicking is the proven
path; the scheduled run is the backup.

## Consequences

- Morning rows are built once and shared (#102/#103); quotes come through
  the provider chain (0017).
- Everything assumes the machine is awake; that constraint is documented
  rather than engineered away (App Store plan, #101, is the eventual out).
