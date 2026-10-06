# Overnight Phase 2a: verified raw-price ingestion

`MassiveOvernightClient.fetch(ticker, start, end)` retrieves one ticker and one
inclusive exchange-date range. It requests Massive custom daily bars with
`adjusted=false`, then the new `/stocks/v1/splits` and `/stocks/v1/dividends`
endpoints filtered by execution and ex-dividend date. The end date must precede
the current New York date so a partial live-day bar is never certified.
Massive's cursor links
omit the original filters, and its [pagination contract](https://massive.com/blog/api-pagination-patterns)
says added filters are ignored. If an action page has `next_url`, the client
bisects that date window and requeries both halves with explicit filters. Only
single-page, filtered action responses certify `verified_none`; a one-day
window that still paginates fails closed. HTTP failure, 403/429, invalid or
adjusted bars, response mismatch, incomplete pagination, or out-of-range
action records fail the whole fetch. The request/page cap is 100 per endpoint.

The resulting `history` has raw OHLCV indexed by midnight XNYS session dates.
Its same-index `provenance` has the Phase 1 fields `source=massive`,
`basis=raw`, `action_source=massive_actions`, and `corporate_action`, plus event
IDs, the verified date window, and fetch times. A date with both a split and a
dividend has status `split` and retains both IDs. The Phase 1 builder excludes
either action status from training. `OvernightPriceStore` writes both frames
atomically into one dedicated per-ticker Parquet at `data/overnight_prices/`,
or an explicitly supplied test/data directory. It never modifies
`data/prices/` or `data/features/`. The store validates finite coherent OHLCV,
XNYS sessions, matching action status/IDs, and timezone-aware fetch times.
Existing ticker files require `overwrite=True`, and even an explicit overwrite
must preserve every already stored session.

Example for a deliberately small, single-ticker fetch:

```python
from datetime import date
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient
from stock_picker.storage.overnight_price_store import OvernightPriceStore

batch = MassiveOvernightClient().fetch("AAPL", date(2026, 9, 28), date(2026, 10, 2))
OvernightPriceStore().write("AAPL", batch.history, batch.provenance)
```

The API key uses the existing `polygon_api_key()` lookup. Without a verified
paid-tier coverage boundary, the client conservatively refuses requests older
than 700 days; callers may set a more recent `action_coverage_start`, but
cannot extend that boundary backward. This slice does not run a backfill or
train a model. A later controlled backfill must confirm action-feed entitlement
before extending this conservative range,
inspect exclusions from the Phase 1 builder, and avoid overwriting existing
overnight data without review. An empty or missing bar remains absent; it is
not turned into a fabricated session. Only the dates of returned bars have
per-bar evidence, so consumers still need the Phase 1 XNYS adjacency check.

Bounded read-only probe on 2026-10-06: AAPL bars for 2026-09-28 through
2026-10-02 returned HTTP 200, five rows, and `adjusted=false`. One split page
and one dividend page returned HTTP 200, one row each, with `next_url` present;
the initial probe did not traverse those pages or persist data. The action
implementation reissues smaller explicitly filtered windows when pagination
would otherwise be required. No key value was printed.

Current provider references:

- [Massive custom daily bars](https://www.massive.com/docs/rest/stocks/aggregates/custom-bars)
- [Massive stock splits](https://massive.com/docs/rest/stocks/corporate-actions/splits)
- [Massive stock dividends](https://www.massive.com/docs/rest/stocks/corporate-actions/dividends)
- [Phase 1 provenance contract](overnight-phase1-data-contract.md)
