# 0005: Repository-pattern stores: parquet for series, SQLite for event logs

Date: 2026-09-04 (backfilled 2026-09-26)
Status: Accepted

## Context

Everything persists to local disk; call sites shouldn't care about formats
or paths, and tests need isolation without monkeypatching.

## Decision

One store class per concern (UniverseStore, PriceStore, FeatureStore,
ModelStore, TradeStore, ScanStore, PaperBookStore, NewsStore,
PrunedFeatureStore, TrainingRunStore, TrainingConfigStore). Columnar series
(prices, features) are parquet; append-style event logs (trades, scans,
paper book, news) are SQLite. Every constructor takes an optional
`data_dir` -- `Store(data_dir=tmp_path)` is the established DI pattern for
testing anything that touches persistence.

## Consequences

- No ORM, no server, no migrations beyond ad-hoc column adds.
- New persistence = new store class following the same shape, not a new
  mechanism.
