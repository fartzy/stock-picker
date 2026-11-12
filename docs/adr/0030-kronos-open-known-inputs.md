# 0030: Pinned Kronos forecasts as ordinary Fit and Rank inputs

Date: 2026-11-12
Status: Accepted

## Decision

The isolated daily Kronos worker supplies four independently selectable/prunable
inputs: `kronos_close`, `kronos_high`, `kronos_low`, and `kronos_range`. They are
point-price forecasts relative to the observed target-session open. They are not
profit probabilities, a new trading policy, or a reason to exclude a ticker.

Data flows from existing PriceStore OHLCV through the pinned JSON-lines worker
into `data/kronos/forecasts.sqlite3`, then into the normal feature and training
pipeline. SQLite stores each immutable runtime/manifest profile once, immutable
request-key forecast revisions, original generation/collection timestamps,
source hashes, raw invalid forecasts, and retryable collection status. Same-key
conflicting outputs fail. A rerun preserves the original observation. Bulk cache
and downloaded weights stay ignored; compact reports remain versioned.

Only the pure `tools/kronos/contract.py` enters the app's Bazel dependency graph.
Torch and all runtime-specific libraries remain in `.runtime/kronos/`. The
worker is loaded once per collector invocation and reused across batches. No
CPU fallback is allowed; a changed execution profile needs a separate cache
and explicitly retrained consuming artifacts.

## Timing, prices and missing values

Every request uses exactly 128 consecutive completed XNYS bars strictly before
its next-session target. Target/future high, low, close and volume never enter
the request or four feature formulas. Each forecast price is first divided by
its saved last context close, then multiplied by a compatible current prior
close and divided by the actual open. The range uses `(high - low)` similarly.

Historical joins require the exact current input hash. Source corrections or
uniform split rescalings make an older request unavailable until recollected;
no stale absolute forecast is compared with a differently scaled open. Known
target split-action rows stay unknown. Live use additionally requires quote and
stored prior closes to agree. Existing PriceStore prices retain the honest
`price_store_ohlcv_as_stored` label: their adjustment convention is not
independently certified, and most source files lack action columns. Unknown
corporate actions cannot be inferred from target outcomes without violating
the open-only contract. No target close/adjusted-close field is used for that.

Reconstructed forecasts retain their real `generated_at` and reconstruction
flag. They are permitted for historical research/training and explicit What If
replay or simulated Test Run. Those callers opt into counterfactual mode; replay
scan IDs and saved model IDs remain separate from the actual morning scan, and
the backend logs the mode. Counterfactual mode cannot publish actual morning
observations. Real morning scoring requires generation before the target open
and before the scoring clock. Missing, invalid, stale, wrong-profile or
incompatible-basis inputs remain NaN; yesterday's K inputs are always cleared.
CSQR and MRCY remain ordinary universe members.

## Readiness and artifact pairing

The normal feature rebuild adds these columns only after every active
ticker/session slot in its retained year has a terminal collection status and
every session has at least one valid forecast. Missing context and invalid
forecasts are explicit terminal results, not zeroes. Worker failures are
retryable and do not satisfy readiness. This is a historical collection gate,
not a performance or promotion test. A three-day pilot cannot unlock columns.

Feature Parquet metadata carries the exact execution fingerprint, manifest
hash, feature formula version and price basis. Pooling rejects mixed/missing
contracts. Every saved consuming model, including the paired SVM source models,
pins that contract and checks it at scoring. Integration tests cover the real
Parquet → pooled dataset → both SVM fold paths → saved Fit/Rank artifacts.
The shared live base rows receive K inputs separately for each saved head, so
different Fit and Rank contracts cannot reuse the wrong values. Old artifacts
with no K consumers remain compatible.

The overnight scenario path also resolves its saved Fit, Rank, source SVMs
and named morning variants separately, using ephemeral price context. Historical
overnight folds retain the same training-frame provenance; they never substitute
a current live cache contract for their saved one.

After a complete authorized backfill and normal feature rebuild, subsequent
scheduled training can consider the four inputs through existing selection
and pruning. This change does not train, activate or replace production models.

## Operations

Run from the repository root. Relative output paths remain rooted there under
`bazelisk run`; the child interpreter does not inherit Bazel's Python path/home
isolation settings.

Start the worker directly (setup and pinned assets: `tools/kronos/README.md`):

```sh
.runtime/kronos/venv/bin/python -m tools.kronos.worker --device mps
```

Bounded collection uses one persistent worker and resumes exact observations:

```sh
bazelisk run //python/stock_picker/ingestion:kronos -- \
  --start 2026-11-10 --end 2026-11-12 --tickers HURN CSQR TTMI MRCY \
  --device mps \
  --expected-fingerprint da7f49ffe7ac8e74363ba177cff10a81e390950aabafa8216c12ae5fa1b3c989 \
  --cache-dir .runtime/kronos/pilot-cache --batch-size 4 --max-seconds 180 \
  --report research/kronos_runtime_2026_11_12/collector_pilot.json
```

The complete retained-year collection, **after verification**,
uses the default cache and active universe (omit `--tickers`):

```sh
bazelisk run //python/stock_picker/ingestion:kronos -- \
  --start 2025-11-12 --end 2026-11-12 --device mps \
  --expected-fingerprint da7f49ffe7ac8e74363ba177cff10a81e390950aabafa8216c12ae5fa1b3c989 \
  --max-seconds 21600 --report research/kronos_runtime_2026_11_12/backfill.json
```

Repeat a stopped command to resume. `--max-requests` bounds inference work.
`--refresh-inputs` explicitly reconciles completed old-history source revisions.
The manual daily command does no training:

```sh
bazelisk run //python/stock_picker/ingestion:kronos -- --daily --max-seconds 600
```

Nightly invokes the same daily collector immediately after price refresh and
before feature rebuilding. It prioritizes the newest five targets within its
budget, then catches up missing sessions across its retained
collection span and automatically rechecks the latest five target sessions for
source revisions. Missing contexts remain retryable after later price refreshes.
A clean clone without the optional runtime/configuration continues normally.
Timeouts, protocol failures and counts appear in backend logs; one transport
restart is allowed. Protocol reads and writes are bounded, including partial
lines. Collection's work limit is checked between requests; worker startup and
one in-flight batch/restart can extend it by their bounded 120-second timeouts.

## Measured pilot scope

The approved pilot covered four tickers × three targets, November 10–12:
8 valid forecasts, 1 preserved invalid TTMI November 11 bar, and 3 missing
CSQR contexts. Total collector time was 3.08 seconds. A repeat took 2.12 seconds,
with 9 cache hits and zero inference. The MPS fingerprint matched the pinned
contract above. HURN's historical inputs matched the shared pure formula;
actual-morning use correctly rejected the reconstructed record. CSQR stayed
NaN. The retained-year readiness gate remained false. Reports:

- `research/kronos_runtime_2026_11_12/collector_pilot.json`
- `research/kronos_runtime_2026_11_12/collector_pilot_resume.json`
- `research/kronos_runtime_2026_11_12/collector_pilot_verification.json`
- `research/kronos_runtime_2026_11_12/coverage_audit.md`

This measured integration and resume behavior, not predictive improvement.
No full backfill, source-price repair, feature snapshot write or production
model training/activation was performed by the pilot.
