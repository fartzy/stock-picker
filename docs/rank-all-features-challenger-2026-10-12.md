# Rank all-feature challenger — 2026-10-12

## Outcome

A second **research-only** ListFold Rank model uses all 19 new price-pattern
features from the [feature-family screen](feature-family-screen-2026-10-12.md).
The 125-input artifact is `data/models/research/rank_all_2026-10-12.pkl`.
The live Rank/Fit models and serving selection were not changed.

This model is **not promoted**: although its ticker-holdout results improve,
top-five returns are worse than the matched baseline in two of three
chronological folds. The [price-breadth-only challenger](rank-breadth-challenger-2026-10-12.md)
also remains research-only.

## Matched evaluation

Both baseline and all-feature challenger trained for 100 LightGBM rounds on
the same 1,799 tickers and 25 post-registry sessions (September 8–October 12;
44,863 rows). The challenger alone added all 19 columns. Each test date in
the three expanding folds is later than its training dates. Top-five is mean
open-to-close return of selected names, not actual broker P&L.

| Test dates | Baseline Rank IC | All-feature Rank IC | Baseline top-five | All-feature top-five |
| --- | ---: | ---: | ---: | ---: |
| Sep 17–24 | 0.0701 | 0.0954 | +0.373% | +0.222% |
| Sep 25–Oct 2 | 0.0975 | 0.0987 | +0.352% | −1.202% |
| Oct 5–12 | 0.0510 | 0.0913 | −1.609% | −1.126% |

The deterministic 200-ticker holdout (5,000 rows) shares the same calendar
dates as training tickers and has already informed earlier feature-family
research. It is not an untouched or future-date validation set:

| Model | Rank IC | Top-five mean return | Top-five hit rate |
| --- | ---: | ---: | ---: |
| Baseline | 0.3124 | +1.247% | 73.6% |
| All 19 features | 0.3197 | +1.366% | 77.6% |

The 19 columns remain gated out of automatic live retraining. No
post-selection future sessions are available yet. Reproduction command:

```sh
bazelisk run //python/stock_picker/training:feature_challenger -- --variant all --rounds 100 --folds 3
```

The command refuses to overwrite an existing artifact; use a separate output
directory for a repeat experiment.

## Full-history correction — 2026-10-13

The saved research artifact above was trained on only 25 recent sessions. To
test the combined model the user intended, we also compared baseline and
all-19 Rank models trained on **all available history** from October 13, 2025
through October 12, 2026: 442,616 training-ticker rows. Older breadth/path
values stay missing, since the point-in-time universe registry begins only in
September 2026. Evaluation still uses the same three strictly later,
post-registry date blocks and the morning Rank list's actual top **20**.

| Test dates | Baseline Rank IC | All-19 Rank IC | Baseline top-20 | All-19 top-20 |
| --- | ---: | ---: | ---: | ---: |
| Sep 17–24 | 0.0514 | 0.0660 | −0.126% | −0.087% |
| Sep 25–Oct 2 | 0.1704 | 0.1768 | +0.297% | +0.606% |
| Oct 5–12 | 0.1310 | 0.1021 | −0.064% | +0.150% |

Top-20 return improved in all three folds, while mean fold Rank IC slipped
slightly from 0.1176 to 0.1149. On the already-consulted ticker holdout,
Rank IC improved from 0.2202 to 0.2837, top-20 mean return from +0.378% to
+0.692%, and top-20 hit rate from 57.4% to 62.8%. That holdout shares dates
with training and is **not** independent future evidence. Performance after
October 12 remains unknown.

The 125-input full-history model is saved separately as
`data/models/research/rank_all_full_2026-10-12.pkl`. It did not overwrite the
live model or the earlier 25-day research artifact. Reproduce the comparison
without writing a model:

```sh
bazelisk run //python/stock_picker/training:feature_challenger -- --variant all --scope full --rounds 100 --folds 3 --evaluate-only
```
