# Outstanding work check-in for November 11 2026

The requested [active price-pattern feature document](../../active-price-pattern-features-2026-11-11.md) records every live opening-pattern and dip input, explains the user's DDD/open-down and UUDD/open-down examples, and distinguishes current model inputs from inactive research candidates. Its 57 opening-pattern and 19 dip entries were checked against the saved live-model inventory. All relative documentation links were checked.

## Scope and logical branches

The starting branch was `research/intraday-context-20261111`, two commits ahead of `origin/main` at `bd5790b72f6975ceb1fb47d96cba067492c98366`. The working tree contained 3,998 outstanding data files: 1,998 feature files; 1,986 price files; one weather file; four model artifacts; MLflow and training-run records; four trade files; two Jev databases; and the paper book.

The branches form an ordered stack. Each branch adds one logical commit to its predecessor; the last branch contains the complete check-in. They are not independent replacements for one another. No old source implementation or stale data stash is reapplied.

| Order | Branch | Scope |
| --- | --- | --- |
| 1 | `fix/what-if-loading-20261111` | Existing UI fix: request cancellation/timeouts, bounded history loading and collapsed Jev results. |
| 2 | `research/intraday-context-20261111` | Existing research: literature archive, reproducible intraday context comparison, raw source evidence, shared experimental formulas and tests. |
| 3 | `docs/active-pattern-features-20261111` | Requested feature inventory and captured live model inputs/composition. |
| 4 | `data/price-weather-snapshot-20261111` | Raw price histories and weather. Most prices end November 10; weather ends November 9. |
| 5 | `data/feature-snapshot-20261111` | Generated feature snapshots, with actual per-file coverage recorded. |
| 6 | `data/model-training-snapshot-20261111` | Current and archived models, training runs and MLflow. No model retraining during check-in. |
| 7 | `data/broker-ledger-20261111` | SQLite/CSV ledger and November 10–11 broker confirmations. |
| 8 | `data/forecast-outcomes-20261111` | Jev forecasts/outcomes and paper-book history. |
| 9 | `data/runtime-archive-20261111` | Normally ignored durable runtime data, consistent database backups and this audit. |

## Validation

[data-validation.json](data-validation.json) records SHA-256, size and read validation for all 3,998 original outstanding files. Every changed Parquet table was read and validated, including 494,890 feature rows, 967,582 price rows and 3,235 weather rows. Coverage is preserved as observed, including files ending before November 10; this check-in did not manufacture missing sessions.

[trade-validation.json](trade-validation.json) verifies 1,781 SQLite rows against 1,781 CSV rows, no duplicate execution keys, and all 51 November 10–11 confirmed broker orders present exactly once. The new training archive and current Fit pickle have identical SHA-256; the live API successfully loaded the Fit and Rank model inventories (158 and 151 inputs respectively).

Seven consistent SQLite backups passed integrity checks and lossless decompression checks. The [runtime snapshot manifest](../../../data/snapshots/2026-11-11/manifest.json) records 104 captured artifacts: 97 files plus seven compressed database backups. The archives preserve committed WAL data without replacing or pausing the live stores.

The research commit's four affected Bazel test targets passed, including temporal leakage, individual feature selection, retry/resume and archive-integrity tests. The preceding UI commit passed TypeScript checking and the requests, What If, overnight What If and Jev display tests. No functional UI/model source changed after those verified commits during this publication. The new documentation and capture script were checked separately.

Official Parquet API references used for the capture script: [read_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html) and [Table.validate](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.validate).

## Older local work

[branch-audit.json](branch-audit.json) compares local heads with remote main, exact merged pull-request heads and the previous reconciliation archive. No additional unresolved historical branch was identified. All four extant secondary worktrees were clean. Stale prunable worktree records were left intact.

The same five stash objects remain unchanged and are already inventoried/preserved in the [November 9 reconciliation](../local-checkin-2026-11-09/README.md). Applying them would reintroduce superseded implementations or older data, so they remain preserved as historical evidence.

## Publication boundaries

The snapshot captures the recorded times, not future activity. New scoring, trades or forecast requests can change live files after capture. Credentials, `.env` files, dependency/build caches, bytecode, locks and SQLite sidecars are not repository publication data. Durable data normally excluded by `.gitignore` is preserved under `data/snapshots/2026-11-11/`.

This check-in preserves the live models and does not promote the six experimental intraday context features. The source archive and trial results document that research separately.
