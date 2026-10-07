# Close-conditioned next-open feature comparison — 2026-10-06

This is a controlled **12-ticker exploratory run**, not evidence that a
midday hold decision works. Verified raw Massive bars covered 2026-06-01
through 2026-10-05 for STLA, WERN, WDC, ALHC, BMY, LQDA, NKTR, TH, CTVA,
MICC, CPB, and RITM. The target is next-session open divided by today's
**actual close**, minus one. Historical Fit/Rank/SVR/SVC outputs were produced
by same-day models fitted on earlier dates. For speed, these same-day models
used the 12-stock subset, not their normal full universe; relative Rank
scores and sparse long-window/SVM inputs can therefore behave differently.

The experiment compared identical rows and folds, using LightGBM's same
`regression_l1` settings (80 boosting rounds) for every variant:

| Variant | Inputs | Fold 1 gap MAE | Fold 2 gap MAE | Combined gap MAE | Combined open MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Unchanged price | predicted gap = 0 | 1.2246% | 1.3348% | **1.2775%** | **$1.4773/share** |
| Core 2 | assumed close / today's open − 1; same-day Fit predicted return | 1.2246% | 1.3513% | 1.2854% | $1.4849/share |
| Existing 13 | original price/scenario + four morning-model outputs | 1.2412% | 1.3583% | 1.2974% | $1.5068/share |
| Expanded 15 | existing 13 + five-session prior return + Fit prediction minus assumed day return | 1.2400% | 1.3612% | 1.2982% | $1.5044/share |
| Weighted 15 | expanded 15, with 3× split-gain priority on the two core inputs | 1.2391% | 1.3466% | 1.2907% | $1.4927/share |

Fold 1 trained through July 17 and tested 324 rows from July 20–August 25.
Fold 2 trained through August 25 and tested 299 rows from August 26–October 2.
Thus 623 ticker-sessions were held out; 909 verified rows had historical
same-day scores in all. The existing-13 control reproduced the first trial's
fold errors exactly. Of the source candidates, 72 lacked warm-up history, 63
crossed a verified dividend event, and 12 final bars lacked an observed next
open. These were excluded rather than treated as valid zero-gap examples.

**Result:** the user's two-input hypothesis is the best *model* variant on
gap MAE and nearly ties the unchanged-price baseline in fold 1, but loses to
it in fold 2 and overall. Giving the two inputs more split priority improves
the 15-input model versus the unweighted 15, but still does not beat the
unchanged-price baseline. Adding the two new features alone does not help.
The 15-input feature contract is retained for further inspection; this run
did **not** replace a live model artifact or claim a trading edge.

For a concrete held-out example, STLA closed at $4.40 on October 2. The
next observed open was $4.46. Core 2 predicted $4.4028; weighted 15
predicted $4.4021; unchanged-price predicted $4.40. All three missed most
of the six-cent rise. On the same date LQDA closed at $28.64 and next opened
at $28.07; core 2 predicted $28.5854 and weighted 15 $28.6081, both a little
closer than the unchanged-price guess but far from the actual open.

Every held-out predicted gap/open and actual gap/open is written by
`overnight_experiment.py` to `held_out_predictions.csv`, with one fold summary
per variant in `fold_summary.csv`. The run's local files are kept outside the
public repository because they contain licensed raw market prices. The
experiment uses documented LightGBM `feature_contri` split-gain weighting;
it does **not** multiply the input price values.

Next validation should use full-universe historical same-day outputs and
more tickers/dates, including a genuine ticker holdout. An honest *midday*
test additionally requires timestamped historical intraday prices; treating
the final close as a known 1:00 PM price would be lookahead. The current
L1/median objective produced gaps close to zero and did not capture large
overnight moves; costed hold-versus-sell analysis remains outstanding.
