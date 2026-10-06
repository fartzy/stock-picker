# Overnight Phase 1 data contract

The overnight label is `Open_(next actual XNYS session) / Close_t - 1`. The same
pure transform builds historical and hypothetical-close features. `t+1` is the
next **observed** bar only when `exchange_calendars` confirms it is the next
XNYS session; otherwise the row is excluded. The final bar remains unlabeled.
The live next-session helper returns unknown when the calendar cannot answer.
Bar indices must be timezone-naive **exchange-session dates at midnight**;
UTC timestamps and intraday timestamps are rejected rather than silently
reinterpreted as New York sessions. Artifact metadata pins XNYS and the
`exchange_calendars` version as well as source, basis, and feature order.

## Required evidence before Phase 2

Each daily bar must have separate, same-index provenance with `source`, `basis`,
`action_source`, and `corporate_action`. The selected `PriceContract` records
the exact source and raw basis. Every bar used by a feature window or label
must match that contract, and each action status must be either `verified_none`,
`split`, or `dividend` based on an actual action-feed check. Split and
ex-dividend windows are excluded; missing verification is also excluded. The
builder reports exclusion counts and dates; it does not infer provenance from
`Adj Close == Close` or discard a large gap merely because it is large.

The hypothetical-close builder separately requires verified provenance for
**today's observed open** (`source`, `basis`, action source/status) and an
explicit raw-basis assertion for the user's assumed close. It refuses missing,
adjusted, or mismatched current inputs. No production caller is wired yet;
Phase 4 must establish that the chosen live-open feed uses the same raw-price
semantics as the historical daily source before supplying that evidence.

The current `PriceStore` writes only OHLCV frames and does not retain this
evidence. For example, the existing WDC parquet has `Open`, `High`, `Low`,
`Close`, `Adj Close`, `Volume`, and `Date` but no source, basis, or action status.
Therefore it is **not** a certified input to Phase 2, even if it contains
plausible prices. No model should be trained by passing fabricated provenance.

The shortest path to a usable research dataset is a separate, provider-tagged
backfill of Massive daily aggregates requested with `adjusted=false`, plus
Massive split and dividend events covering the entire requested date range.
Persist the raw-price source/basis and each date's action-screen result beside
the bars, then pass those records through this builder and inspect exclusions.
An alternative is to backfill source/action evidence for the existing Yahoo
rows and verify every raw Open/Close against the provider's documented basis;
column names and `auto_adjust=False` alone are insufficient. Neither backfill
is performed in Phase 1. The separate single-ticker implementation is described
in [Phase 2a ingestion](overnight-phase2a-ingestion.md); it does not run a
backfill or train a model.

The first feature set is exactly the nine ordered columns in
`training/overnight_dataset.py`: six completed prior closes supply yesterday's
close, one prior return, and five-return volatility; today's observed open and
its gap; an explicitly assumed close and returns derived from only that
assumption; and weekday. No same-day final high, low, volume, VWAP, news, or
morning `overnight_gap` enters this frame. Later overnight artifacts must
persist `PriceContract.artifact_metadata()` and the exclusion report.

Primary documentation checked on 2026-10-06:

- [exchange_calendars XNYS/session APIs](https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md)
- [yfinance `download` auto-adjust and actions parameters](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Massive split-adjustment semantics](https://polygon.io/knowledge-base/article/is-polygons-stock-data-adjusted-for-splits-or-dividends)
- [Massive stock splits API](https://polygon.io/docs/rest/stocks/corporate-actions/splits)
- [Massive stock dividends API](https://www.massive.com/docs/rest/stocks/corporate-actions/dividends)
