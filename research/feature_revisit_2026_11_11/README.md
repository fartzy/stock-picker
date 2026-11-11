# Morning feature review — November 11, 2026

Status: comparison running. No production model, pruning setting or UI changed.

- 443,609 stock-days, 1,799 stocks, 250 sessions. MRCY contributes all 250 sessions.
- Baseline: 151 raw inputs. Reconsider 93 exclusions; the two known bad dip ratios stay excluded.
- New candidates: count, standard deviation, age, shrunk mean and recent mean for each two/three/four-day U/D sequence plus opening-gap bucket: 15 inputs.
- Evaluate Rank and raw LightGBM return regression separately. The latter is **not** production Fit with its seven SVM outputs.
- Same-day open-to-close target; zero assumed expenses. Rank #1, top five and top 20 reported separately.

## Rules fixed before results

Three 21-session selection blocks: July 15–August 12, August 13–September 11, September 14–October 12. Each fit uses only preceding dates. October 13–November 10 is held out from this iteration's selection. These historical dates were examined in earlier research; this is not fresh forward evidence.

Every eligible exclusion participates in an all-restored fit. Each fold separately selects the five highest-gain excluded columns using fitting data, tests them individually, and tests the top-five/top-ten sets and restored seasonality group. Each new input is tested individually, by statistic, and together. This does not individually test every one of the 93 older exclusions.

Select one variant per model: positive mean Rank #1 return lift, improvement in at least two of three blocks, and no mean top-five deterioration. Highest qualifying mean Rank #1 lift wins. If none qualifies, retain baseline. Freeze selection before the later-date comparison. No automatic production promotion.

## Feature timing

Shared builder: [pattern_evidence.py](../../python/stock_picker/features/pattern_evidence.py). Features use strictly earlier outcomes. Missing exchange sessions break sequences. Dispersion requires two observations; unknown patterns remain unknown. Shrinkage adds 20 observations at the ticker's strictly prior unconditional mean. Recent means decay with a fixed 63-session half-life. These constants are experimental choices, not fitted optimal values.

Tests cover hand-calculated statistics, missing outcomes/sessions, current/future-price mutation, live/replay parity, deterministic top-one results and date/selection boundaries.

## Repeat

```sh
bazelisk build //python/stock_picker/training:feature_revisit
bazel-bin/python/stock_picker/training/feature_revisit --output research/feature_revisit_2026_11_11 --phase prepare
bazel-bin/python/stock_picker/training/feature_revisit --output research/feature_revisit_2026_11_11 --phase screen --model rank
bazel-bin/python/stock_picker/training/feature_revisit --output research/feature_revisit_2026_11_11 --phase confirm --model rank
```

Use `--model fit_raw` for the separate regression screen. Use a new output directory for a later review. Resume interrupted screens in their existing directory; frozen protocol/selection files cannot be overwritten. Source or panel hash changes fail closed.

`protocol.json` records all source-data hashes, formulas, dates, pruning reasons, package versions and model settings. The 810 MB panel is a local ignored cache rebuilt from the versioned inputs. Predictions, daily outcomes and compact reports are preserved per variant. The active universe and revised provider bars limit historical interpretation; include untraded candidates but do not claim a historical point-in-time universe.
