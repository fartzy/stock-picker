# Rank price-breadth challenger — 2026-10-12

## Outcome

A new **research-only** ListFold Rank model was trained with the three
price-breadth features selected in the earlier [feature-family screen](feature-family-screen-2026-10-12.md).
Its artifact is `data/models/research/rank_breadth_price_2026-10-12.pkl`.
It has 109 inputs, including `market_advance_fraction`,
`market_above_sma_20d_fraction`, and `market_return_dispersion_1d`.
Neither the live Rank/Fit artifacts nor the serving selection was changed.

The model is **not promoted**. The 100-round comparison is mixed, and the
latest chronological fold still loses money on top-five picks.

## Matched evaluation

Both models use the same existing features, 100 LightGBM rounds, 1,799
training tickers, and the same 25 post-registry sessions (September 8–October
12; 44,863 rows). Only the challenger adds the three price-breadth columns.
Three expanding chronological folds keep each test date after its training
dates. Top-five return is the mean open-to-close return of selected names,
not a realized or compounded trading result.

| Test dates | Baseline Rank IC | Challenger Rank IC | Baseline top-five | Challenger top-five |
| --- | ---: | ---: | ---: | ---: |
| Sep 17–24 | 0.0701 | 0.1054 | +0.373% | +1.440% |
| Sep 25–Oct 2 | 0.0975 | 0.0872 | +0.352% | +0.150% |
| Oct 5–12 | 0.0510 | 0.0877 | −1.609% | −0.800% |

The deterministic 200-ticker holdout (5,000 rows) was never fit, but shares
calendar dates with training tickers. It was also consulted in the earlier
40-round screen, so this 100-round check is not a fresh independent holdout
or a substitute for future-date validation:

| Model | Rank IC | Top-five mean return | Top-five hit rate |
| --- | ---: | ---: | ---: |
| Baseline | 0.3124 | +1.247% | 73.6% |
| Challenger | 0.3128 | +1.291% | 72.8% |

The unchanged research-feature gate continues to exclude all 19 candidate
columns from automatic live retraining. The other 16 new columns were not
included in this model because the earlier family screen did not support
their promotion. No post-selection future sessions are available yet.
Reproduction command:

```sh
bazelisk run //python/stock_picker/training:feature_challenger -- --variant breadth_price --rounds 100 --folds 3
```

The command refuses to overwrite an existing artifact; use a separate output
directory for a repeat experiment.
