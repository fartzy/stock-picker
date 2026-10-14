# Price-pattern feature-family screen — 2026-10-12

## Decision

Persist all 19 new candidate columns, but withhold them from automatic training
for now. Price-based market breadth is a **provisional Rank candidate**, not a
promoted production feature. No live model or per-run selection was changed.

The universe registry only begins in September 2026. Breadth and trajectory
features therefore mask dates before each ticker's `first_seen`, instead of
projecting today's members backward through the full year. This leaves too
little recent history for a confident production decision.

## Method

`training/feature_family_search.py` compares the same baseline feature set
against one family/subfamily at a time. It uses 40 LightGBM rounds, three
chronological folds, deterministic exclusion of 10% ticker holdout, and no
model/configuration writes. Fit is represented by a single LightGBM return
model, not the production blend; Rank uses the repository's ListFold rank
objective. Top-five returns are mean open-to-close returns of selected names,
not compounded portfolio returns or actual broker P&L.

Full-history volatility screen: 251 dates, 442,616 training-ticker rows.
Volatility candidate coverage was 95.9% of these rows. The complete family
was essentially neutral versus baseline Fit MAE across folds:

| Fit proxy | Fold 1 MAE | Fold 2 MAE | Fold 3 MAE | Fold top-five mean returns |
| --- | ---: | ---: | ---: | --- |
| Baseline | 1.861% | 1.834% | 1.616% | +0.09%, +3.78%, +0.30% |
| All new volatility | 1.857% | 1.839% | 1.616% | +0.23%, +3.76%, +0.29% |
| Volatility shape only | 1.854% | 1.842% | 1.613% | +1.05%, +3.57%, +0.24% |
| Up/down volatility only | 1.868% | 1.837% | 1.617% | +0.15%, +3.90%, +0.53% |

Recent point-in-time screen: 25 dates from September 8 to October 12,
44,863 training-ticker rows. Volatility/breadth/trajectory coverage was
99.96%/99.99%/96.47% within this restricted interval. The most consistent
Rank result was price-based breadth (advance share, share above 20-day average,
return dispersion; excluding up-volume):

| Rank proxy | Fold Rank IC | Fold top-five mean returns |
| --- | --- | --- |
| Baseline | 0.0565, 0.0972, 0.0376 | −0.30%, +0.02%, −2.74% |
| Price breadth | 0.0955, 0.1031, 0.0696 | +0.74%, +0.50%, −2.00% |

The last fold still lost money on its top five. Fit did not confirm the same
top-five improvement: its baseline was +0.95%, −0.10%, −0.66%, versus
+0.83%, −1.39%, +0.18% with price breadth. Trajectory and the combined
family were mixed on Rank IC and/or top-five returns; neither is selected.

After selecting **price breadth only** from the chronological folds, one
ticker-holdout check trained on 1,799 other tickers and evaluated 200 never-fit
tickers (5,000 rows, the same 25 calendar dates):

| Rank proxy | Rank IC | Top-five mean return | Top-five hit rate |
| --- | ---: | ---: | ---: |
| Baseline | 0.2646 | +0.876% | 63.2% |
| Price breadth | 0.2709 | +0.913% | 68.0% |

The relative holdout improvement is small. Since held-out tickers share dates
with training tickers, this does **not** establish future-date or live trading
performance. No other candidate was tried on the ticker holdout after this.

## Next evidence required

Accumulate more post-registry sessions, then repeat the predeclared Rank
price-breadth comparison on future dates. Rank-specific inclusion must also be
separated from Fit's inclusion before promoting a Rank-only winner; the
current training configuration applies one positive selection to both.
