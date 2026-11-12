# Kronos input coverage audit — 2026-11-12

Local data only; completed prices through **2026-11-11**, daily target **2026-11-12**. All **2,000 active universe tickers** are included, including untraded candidates and holdout stocks. This measures valid inputs; generated forecasts can still fail output validation.

| Scope | Candidate ticker-target rows | Eligible rows | Eligible tickers | Estimated MPS inference only |
| --- | ---: | ---: | ---: | ---: |
| Actual retained feature target rows, 2025-11-12–2026-11-11 | 494,940 | **484,935** | 1,971 | **186.862 min (3.11 h)** |
| Full 2,000 × 251-session grid over the same period | 502,000 | 484,951 | 1,971 | 186.868 min |
| Daily target 2026-11-12 | 2,000 | **1,878** | 1,878 | **43.419 sec** |
| All allowed historical target dates from 2024-07-01 | 1,190,000 | 715,064 | 1,971 | 275.538 min |

The actual retained target grid uses existing PriceStore dates under the exact retention predicate `date > 2025-11-11 and date <= 2026-11-11` (`python/stock_picker/features/main.py:25-33`). Its 10,005 unavailable rows comprise **10,002 short-history**, **1 gap**, and **2 invalid-input** rows; there are **0 missing files** and **0 structural errors**. The full date×ticker grid has 16 otherwise eligible targets without a stored target-day row, listed in the JSON. Those forecasts would have no existing feature row to join. The earliest eligible target anywhere in the available two-year cache is 2025-05-20.

Daily unavailable inputs comprise **29 short histories**, **17 gaps/stale tails**, and **76 invalid-input windows** (122 total). These are mutually exclusive primary reasons, in that order; overlapping evidence includes 29 tickers with leading missing context, 4 with internal gaps, 23 with stale tails, and 81 with invalid bars. Every source price file exists. A missing Kronos input must leave the stock available to the ordinary application models.

The daily context requires **128 consecutive XNYS sessions, 2026-05-12–2026-11-11**. Each OHLC must be positive, finite, coherent, and within float32 range; volume must be nonnegative and finite; OHLC4×volume must also be finite and within range. No target-day value enters its own forecast request (`tools/kronos/contract.py:37-55`, `tools/kronos/worker.py:72-90`).

**CSQR:** its local file contains **84 valid consecutive sessions, 2026-07-16–2026-11-11**. The exact absent leading span is **44 XNYS sessions, 2026-05-12–2026-07-15**. There are no internal gaps, invalid bars, or stale tail within its available daily context. Historical eligible rows: **0**. This is a local-cache fact; no company-history explanation is inferred. If every subsequent session is available and valid, the first possible target with 128 prior rows would be 2027-01-19.

**MRCY:** all **251** retained historical targets and the current daily target are eligible. Its 501 stored rows span 2024-11-12–2026-11-11; 373 historical target dates are eligible across the full cached history. It is included without any trade or holdout exclusion.

**Latest-bar quality:** 81 active tickers each have one invalid bar: **78 on 2026-11-11**, one each on 2026-09-10, 2026-10-05, and 2026-11-06. There are **79 incoherent OHLC bars** and **2 rows with missing/nonfinite numeric input**. Example AD on Nov 11: Open 28.979999542236328, High 28.739999771118164, Low 27.549999237060547, Close 27.75, Volume 299222; Open exceeds High. Current-day inputs correctly reject these windows. Most historical forecasts through Nov 11 remain eligible because Nov 11 actuals are excluded from that day's input.

Validation used the real worker's guards for **203 local windows**, with a sentinel stopping immediately before inference: zero mismatches. A second, independent shifted rolling-window calculation re-read and re-hashed **all 2,000 files**, matching every ticker's snapshot eligibility count. No models loaded, forecasts generated, prices repaired/downloaded, features written, or commits created.

Timing estimates use the measured **0.02312 sec/request** MPS rate and exclude load, construction, serialization, cache/storage, and other pipeline overhead. Source data and runtime code hashes, per-date/per-ticker coverage, unavailable ticker lists, exact missing spans, validation samples, and invalid input values are preserved in `coverage_audit.json`. Historical reconstructions use currently stored histories; this audit does not establish historical first-observed availability.

Source digest: `cf510385c245d15371cccedfd3528cb843241ad1ab92007440048bdbc0f4d8af`.

## Latest-bar provenance investigation

The local evidence shows **latest-day Yahoo/yfinance bars are revised on the next refresh**. The current Nov 11 inconsistencies were persisted after close, so the elapsed-close buffer alone does not establish OHLC validity. This investigation found no app merge that splices individual OHLC fields and no morning-open PriceStore write-back.

The Nov 11 nightly log records Yahoo downloading 2,000 histories at **16:30:43 CST** and writing all 2,000 through Nov 11 by **16:30:47 CST**. Finnhub then attempted 23 missing latest sessions and **wrote zero**. See `/Users/michael.artz/Library/Logs/stock-picker/nightly.log:39503-39505` and `:39529`. All 78 files with an invalid Nov 11 bar have modification times **2026-11-11 22:30:43.360623–22:30:47.850422 UTC**, precisely within the Yahoo write phase; their files were not rewritten by the Nov 12 morning scan.

| Nov 11 row | Open | High | Low | Close | Volume | File modification time UTC |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AD | 28.9799995422 | 28.7399997711 | 27.5499992371 | 27.75 | 299,222 | 22:30:47.315290 |
| AJG | 276.6700134277 | 282.2099914551 | 276.8850097656 | 282.0100097656 | 1,301,582 | 22:30:46.566493 |
| V | 417.2525024414 | 425.7099914551 | 417.2999877930 | 425.5499877930 | 4,512,595 | 22:30:46.171464 |

The comparison read all 2,000 prior PriceStore files from Git object reference `2807a881c659b20ba97534fe127eb4cc42134bf8` without changing Git or the working tree. That version ends on Nov 10 for 1,977 tickers. It contains **76 incoherent Nov 10 latest bars, all 76 corrected in the current Nov 11 refresh**, plus an older missing Open for WBD on Oct 5 that remains missing. Across all prior latest rows, 1,976 changed in at least one OHLCV field: 276 Opens, 750 Highs, 710 Lows, 4 Closes, and 1,966 Volumes. Examples:

- A, Nov 10: Open **179.2299957275** exceeded the previously stored High **179.0500030518**; the new High is **179.3200073242**, making the row coherent.
- AAMI, Nov 10: the prior Low **95.8499984741** exceeded Open **95.3899993896**; the new Low is **95.3799972534**.
- MRCY, Nov 10: even its valid row changed, with Open **79.8199996948 → 80.3300018311**. Reconstructed forecast identities therefore need the exact source-input hash.

The write path is Yahoo daily download with `auto_adjust=False` (`python/stock_picker/ingestion/yfinance_client.py:57-76`) → completed-date filter (`python/stock_picker/ingestion/refresh_prices.py:149-155`) → append/deduplicate by session, retaining the **entire incoming row** (`python/stock_picker/ingestion/refresh_prices.py:67-86`) → PriceStore (`python/stock_picker/storage/price_store.py:41-45`). PriceStore only checks individual OHLC positivity/finiteness; it does **not** check High/Low coherence (`python/stock_picker/storage/price_store.py:15-28`). Finnhub fallback checks whether the latest date is missing, so a present but incoherent Nov 11 row is never retried (`python/stock_picker/ingestion/refresh_prices.py:41-53`, `:101-117`). The morning path reads completed history (`python/stock_picker/training/live_rows.py:86-105`); temporary opening-price feature rows are constructed in memory (`python/stock_picker/features/dip_context.py:261-267`).

The practical issue is **acceptance of unreconciled latest-day bars into PriceStore**, compounded by a fallback that only detects missing dates. Later provider revisions are directly demonstrated for Nov 10; correction timing for the current Nov 11 rows remains unknown until another authorized refresh occurs. Current files have only pandas schema metadata and no row-level provider, retrieval-time, or raw-response provenance, so this audit cannot isolate the upstream Yahoo feed from yfinance's construction of its latest daily row. No refetch or data repair was performed. Keep affected Kronos inputs unavailable and revalidate their complete windows after a later refresh; retain the tickers in ordinary models.
