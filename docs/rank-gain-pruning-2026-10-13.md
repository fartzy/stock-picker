# Small gain-pruning test — 2026-10-13

## Result

We compared the full-history, 125-input ListFold Rank research model with
versions that dropped the bottom 3 or 5 inputs by LightGBM training gain.
All 19 new price-pattern inputs remained in every tested model. Importance
was recomputed **inside each training fold**; no scored date selected its own
features. All three variants used 100 rounds, the same 442,616 training-ticker
rows (1,799 tickers, Oct 13, 2025–Oct 12, 2026), and the same three later
test blocks. These are modeled open-to-close returns, not broker P&L.

| Features | Sep 17–24 top-20 | Sep 25–Oct 2 top-20 | Oct 5–12 top-20 | Mean Rank IC |
| --- | ---: | ---: | ---: | ---: |
| 125 (all) | −0.087% | +0.606% | +0.150% | 0.1149 |
| 122 (bottom 3 removed) | −0.235% | +0.700% | +0.150% | 0.1029 |
| 120 (bottom 5 removed) | −0.235% | +0.857% | +0.642% | 0.1020 |

The five-prune model had the highest equal-fold mean top-20 return (+0.422%
versus +0.223% for all 125), but a lower Rank IC. The five dropped inputs
all had zero training gain and were older news/open-pattern columns; the
specific names differed by fold. On full-history retraining, the saved model
dropped `continuation_open3_seasonality`, `news_article_count_1d`,
`news_has_insider_sell_3d`, `news_has_material_1d`, and
`news_has_material_3d`.

On the 200-ticker, 5,000-row holdout, the five-prune model had Rank IC 0.2856
versus 0.2837 for all 125, but top-20 return **fell** from +0.692% to +0.638%
and hit rate from 62.8% to 61.2%. This holdout shares dates with fitting
tickers and has been consulted in earlier research, so it is not a fresh
future validation set. Removing zero-gain columns can still perturb
LightGBM's feature subsampling; the fold improvement is not proof that those
columns were harmful.

The 120-input model is saved at
`data/models/research/rank_gain_prune_5_full_2026-10-12.pkl` and listed as
**research only** in the Models tab. It was **not** promoted to the live
morning model or automatic retraining. Neither small prune consistently beat
all 125 across the forward blocks and ticker holdout.

Reproduce the forward comparison without replacing a model:

```sh
bazelisk run //python/stock_picker/training:importance_prune_search -- --rounds 100 --folds 3
```

The research script refuses to overwrite an existing saved artifact.
