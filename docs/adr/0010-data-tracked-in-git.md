# 0010: All of data/ is committed; a fresh clone is a working, trained app

Date: 2026-09-07 (backfilled 2026-09-26)
Status: Accepted

## Context

Prices, features, models, trades, scans and the paper book all live under
`data/`. Gitignoring them means a fresh clone is a broken app and the trade
history has no durability story (#64, #65).

## Consequences / Decision

Track it all. Git history doubles as the audit log for morning picks and
trades ("Morning picks" and fill commits). Costs: repo size grows with
parquet churn, and concurrent sessions must pull before branching
(documented in CLAUDE.md). Accepted for a single-owner repo.
