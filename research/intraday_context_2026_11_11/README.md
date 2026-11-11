# Intraday context experiment — November 11, 2026

The first iteration does **not** establish a better probability model. Market context slightly reduces Rank #1 probability error, but the interval includes both improvement and deterioration. Same-clock volume improved in training-period validation and then failed to improve later Rank #1 probability forecasts. No variant beats the existing fit-only time-of-day baseline. The live model is unchanged.

This is a completed research comparison with reusable data collection, not a production feature promotion. All returns use **zero assumed trading expenses**.

## What was tested

Four matched variants use the same 3,679 snapshots, 460 stock-days, 264 tickers and 313 sessions from July 1, 2025 through November 6, 2026:

1. Existing Intraday Entry features.
2. Existing inputs plus SPY return from open, SPY 15-minute return, stock-minus-SPY return from open, and stock-minus-SPY 15-minute return.
3. Existing inputs plus cumulative and five-minute dollar volume relative to the same clock on earlier sessions.
4. Both additions together.

The market difference uses unit beta, not an estimated market beta. Dollar volume is minute volume × minute VWAP. Each ratio uses the median corresponding observation from the **previous 20 exchange sessions**, requires at least ten usable sessions and at least 80% bar coverage, and then takes its natural log. Dollar volume limits mechanical share-count effects around splits. Missing days remain in the calendar lookback; they are not replaced by older successful downloads.

Only completed minutes enter features. The entry stays the exact checkpoint minute's open, and the exit stays the final regular minute's close. No final daily volume, later low or unfinished minute enters the predictors. Calendar handling includes DST and early closes. Historical bars retrieved later may contain revisions; these are not first-seen live observations.

The frozen [protocol](protocol.json) and [shared feature implementation](../../python/stock_picker/features/intraday_context.py) contain the definitions. The implementation is available to future historical/live callers but is not activated in production.

## Results

Fit: July 1, 2025–April 27, 2026. Calibration: April 28–August 5, 2026. Evaluation: August 6–November 6, 2026. Models use identical hyperparameters and date/ticker/checkpoint weighting. Evaluation contains 874 snapshots across 63 dates; **Rank #1 has 446 snapshots across 55 dates**. These dates were already examined in earlier research, so this is a retrospective comparison, not fresh independent confirmation.

Rank #1 probability error uses the Brier score: lower is better. It is **not** a return or win rate. Return MAE is the average absolute error in predicted return, expressed in percentage points.

| Variant | Brier error | Difference versus existing | 95% interval for difference | Return MAE, percentage points |
| --- | ---: | ---: | --- | ---: |
| Existing helper | 0.25640 | — | — | 2.6173 |
| Add market context | 0.25562 | −0.00078 | −0.00408 to +0.00302 | 2.5897 |
| Add same-clock volume | 0.25648 | +0.00008 | −0.00185 to +0.00187 | 2.6143 |
| Add both | 0.25608 | −0.00032 | −0.00294 to +0.00254 | 2.5980 |
| Fit-only time-of-day probability baseline | **0.24584** | — | — | — |

The market variant lowers return MAE by 0.0275 percentage points (about 2.75 basis points of forecast error). Its nominal paired interval excludes zero for this secondary metric, but return RMSE worsens from 4.1927 to 4.2053 percentage points. Multiple comparisons and previously inspected evaluation dates limit this evidence. Better average error alone is not a demonstrated profitable entry rule.

For **Rank #1 observations already below the open**, only 38 evaluation dates contribute. Market context changes probability error by −0.000061, with an interval of −0.00398 to +0.00351. That does not establish a better dip-buying trigger. Across all candidate ranks, adding volume worsens probability error by +0.00200; the paired interval is +0.00056 to +0.00330.

Intervals use 2,000 moving-block bootstrap draws with blocks of five observed dates. Repeated snapshots stay together within date. These are nominal descriptive intervals, not adjustments for all research trials. See [comparisons.json](comparisons.json) for dip and complete-context subsets and [evaluation_predictions.parquet](evaluation_predictions.parquet) for every evaluated prediction.

## Individual-feature screening

Each of the six inputs was also tested individually using three expanding chronological prefixes **entirely inside the outer fitting period**. Duplicate validation dates keep their earliest forecast. The screen covers 83 Rank #1 validation dates, picks at most the two individually lowest Brier errors below the existing model, and never accesses outer calibration/evaluation labels for selection.

| Individual addition | Inner-validation Brier difference |
| --- | ---: |
| Cumulative same-clock dollar volume | −0.001047 |
| Five-minute same-clock dollar volume | −0.000878 |
| Stock-minus-SPY return from open | −0.000696 |
| SPY return from open | −0.000464 |
| Stock-minus-SPY 15-minute return | −0.000002 |
| SPY 15-minute return | +0.001061 |

The screen selected the two volume features. Their combined outer result is the volume row above: no improvement in Rank #1 probability error. This preserves the distinction between training-period promise and later evidence. It does not permanently reject either feature for other models, horizons or larger samples. The complete screen and date boundaries are in [individual_feature_screen.json](individual_feature_screen.json).

## Data coverage and repeatability

The collector retrieved **5,771,289 minute bars in 788 bounded ranges**, with no failed ranges. The 116 MB compressed local source archive retains provider identity, raw-price basis, retrieval timestamp, complete-pagination evidence and checksums. Successful ranges are reused; failed requests retry and remain visible. Existing corrupt archives fail closed rather than being silently replaced.

SPY-from-open features cover every snapshot. Missing counts: SPY 15-minute return 433; relative 15-minute return 974; cumulative same-clock dollar volume 551; five-minute same-clock dollar volume 406. The earliest 09:35 checkpoint cannot have a 15-minute regular-session history. Other missingness includes sparse stock bars and insufficient historical coverage. All four variants retain the same rows and use missing values rather than filtering based on outcomes. See [coverage.json](coverage.json).

There is **no ticker exclusion**, including MRCY. The existing frozen baseline model's cached top-20 dataset contains no MRCY rows. Its November 4 live-model record remains in the separate original live cohort; the two ranking systems have different provenance. Extending the live-model cohort, including MRCY, is a separate coverage step, not a reason to discard it. Lower ranks in this study are selectively cached, so this is not complete top-20 history.

Run from the repository root:

```sh
PYTHONPATH=python .venv/bin/python research/intraday_context_2026_11_11/collect.py
bazelisk run //research/intraday_context_2026_11_11:study -- --screen
```

The collector uses the saved base frame, merges overlapping 45-calendar-day lookbacks and divides them into requests of at most 28 days. The study uses the repository's pinned runtime. `--reuse-frame --screen` reruns fitting/evaluation on the saved context frame without rebuilding features. The collector is for completed prior dates and intentionally refuses today's in-progress date.

The baseline frame and source hashes originate in [the existing dataset builder](../../python/stock_picker/training/intraday_entry_dataset.py). [run_manifest.json](run_manifest.json) records code, protocol and frame hashes. No active model pickle is written. Official API semantics: [Massive custom bars](https://massive.com/docs/rest/stocks/aggregates/custom-bars).

## Verification and next iteration

All four affected Bazel test targets pass, covering completed-minute timing, future-history rejection, sparse-data handling, split share-count invariance, model chronology, opt-in feature pruning, bounded requests, retry, cache reuse and checksum failure. [audit.json](audit.json) confirms the original 3,679 rows are unchanged, every entry and close calculation matches the archived bars, and the active model hash is unchanged.

The next useful iteration is broader **chronological candidate coverage**, followed by news/event-conditioned declines. Capture every selected candidate's original rank/model/time, collect current prices and benchmark bars independently of UI refresh, and reconcile after close. That supplies forward evidence and avoids the current selected-cache limitation. This historical collector is reusable, but it is **not yet a scheduled live collector**.

Keep market-relative movement as a candidate, test features individually, and evaluate a return head separately if appropriate. Do not promote the entire volume family based on its training-period result. Sector context still needs trustworthy historical membership; timestamped news still needs incremental first-seen collection. These collection requirements remain prerequisites for reliable production use.
