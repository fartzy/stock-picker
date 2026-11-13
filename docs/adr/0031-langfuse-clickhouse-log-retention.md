# 0031: Cap local Langfuse ClickHouse self-logging

Date: 2026-11-13
Status: Accepted

## Context

The local Langfuse stack from [0016](0016-news-skip-rules-and-local-tracing.md)
runs ClickHouse 24.8 with stock settings. By 2026-11-13 it held 803
`news_day_judge` traces (~426 KB in the `default` database) but ~2 GB of
ClickHouse's own `system.*` log tables (`text_log` 860 MB,
`asynchronous_metric_log` 409 MB, `trace_log` 369 MB, `metric_log` 347 MB) plus
~880 MB of text logs in the container layer. On a near-full laptop disk, the
tracing sidecar was costing ~3 GB to store under 1 MB of useful data.

## Decision

Mount `langfuse/clickhouse-logging.xml` into `/etc/clickhouse-server/config.d/`:

- Disable `text_log`, `trace_log`, `metric_log`, `asynchronous_metric_log`.
- Keep `query_log`, `part_log`, `processors_profile_log` with a 7-day TTL for
  debugging slow Langfuse queries.
- Server text log at `warning`, rotated at 3 x 50 MB.

`opentelemetry_span_log` is left at defaults: the stock config gives it an
explicit `<engine>`, and ClickHouse rejects a separate `<ttl>` next to an engine
(Code 36 `BAD_ARGUMENTS`, container crash-loops).

## Consequences

- Volume dropped from 2.2 GB to 25 MB; the 803 traces were preserved (existing
  system tables were truncated once by hand, not by this config).
- ClickHouse internal metrics/stack-trace history is no longer available
  locally. Re-enable a table by deleting its `remove="1"` line if ever needed.
- Langfuse trace data itself has no TTL and is unaffected.
