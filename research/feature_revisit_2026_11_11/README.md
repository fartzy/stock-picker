# Morning feature review — November 11, 2026

Status: complete. No production model, pruning setting or UI changed.

- 443,609 stock-days, 1,799 stocks, 250 sessions. MRCY contributes all 250 sessions.
- Baseline: 151 raw inputs. Reconsider 93 exclusions; the two known bad dip ratios stay excluded.
- New candidates: count, standard deviation, age, shrunk mean and recent mean for each two/three/four-day U/D sequence plus opening-gap bucket: 15 inputs.
- Evaluate Rank and raw LightGBM return regression separately. The latter is **not** production Fit with its seven SVM outputs.
- Same-day open-to-close target; zero assumed expenses. Rank #1, top five and top 20 reported separately.

## Results

The selected Rank candidate adds three **pattern-age** inputs: sessions since the last observed matching two/three/four-day sequence and opening-gap bucket.

| Reserved 21 sessions | Baseline | Candidate | Difference |
| --- | ---: | ---: | ---: |
| Rank #1 mean daily return | +2.144% | +2.597% | +0.452 percentage points |
| Rank top-five mean daily return | +0.623% | +0.918% | +0.296 percentage points |
| Rank top-20 mean daily return | +0.220% | +0.174% | −0.046 percentage points |

The paired Rank #1 95% interval is −0.292 to +1.077 percentage points; top-five is −0.085 to +0.767. Both include no improvement. These are exploratory price returns across 21 dates, not a demonstrated live upgrade. The current model remains active.

Raw Fit selected a training-gain-based restoration policy, which restored `spy_overnight_gap` for the reserved-period fit. Its top-one return improved, but top-five fell from +0.839% to +0.437%. This does not justify changing the deployed Fit stack.

All 190 saved prediction files (7,160,584 predictions across variants, **not independent samples**) passed a separate audit: labels equal the frozen panel, every variant covers the same dates/stocks, all fits precede their tests, and daily top-K returns recompute exactly. See [summary.json](summary.json), [per-feature review](prune_review.json), and [audit script](summarize.py).

Next: forward-check the three age inputs; retest eligible exclusions after 21 new completed sessions or a source/model change. The review is repeatable but **not automatically scheduled**. Known bad ratios require repair before restoration. The other pattern inputs remain individually available for later experiments.

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
