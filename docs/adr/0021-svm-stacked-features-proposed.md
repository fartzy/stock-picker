# 0021: Proposed: SVM as stacked LightGBM columns, not a blender member

Date: 2026-09-27
Status: Proposed (research path only — no production columns, no pickle,
no pipeline change until the walk-forward + holdout bar in this record
clears)

## Context

ADR 0020 rejected LinearSVR as a Fit ensemble member: the overnight
stability rerun failed ≥3/4 folds on both acc and rank IC, so it never
reached holdout. Production Fit stays solo LightGBM.

That close-out answers one question: should morning Fit be a weighted
average of a tree forecast and an SVM forecast? No.

A different question is still open: can the SVM's *view* of the same
open-known row help LightGBM if it arrives as extra columns, the way
`pca_1..5` and `cluster_id` already do? Trees are good at using a few
strong linear summaries they cannot easily grow themselves. The
(3,1,0)-vs-(3,0,1) comparison in 0020 is the reason to try this —
epsilon-insensitive loss looked like a different linear lens from ridge,
just not a stable enough second forecast to pay for a nightly blend.

This is stacking (a model output used as a feature), not blending (two
forecasts averaged). Logistic regression in this repo is already the
precedent for "fit a linear model, keep it off the blender" — it is a
coefficient lens today; here the linear model would feed numbers into
LightGBM instead of the importance panel.

## Decision (so far)

Do **not** add SVM columns to `features/pipeline.py` or the parquet
feature store. A stacked column is the output of a fitted estimator; it
does not have a pandas formula, and stuffing it into the three
description/formula/example views (ADR 0007) would be a lie. It lives in
the training layer, generated per walk-forward fold, same as any other
model output.

The candidate handful — three columns, not a kitchen sink:

| column | estimator | what LightGBM sees |
|---|---|---|
| `svr_oof_pred` | LinearSVR (existing `train_svr`) | predicted open→close return |
| `svc_direction_margin` | LinearSVC, label = sign of day-session return | signed distance to the up/down plane (`decision_function`) |
| `svc_gate_margin` | LinearSVC, label = (return > 0.5%) | signed distance to the production buy-gate plane |

Three is the cap for the first search. More SVMs on feature subsets wait
until these three have numbers. Kernel SVR / kernel SVC on the pooled
443k rows is still forbidden (ADR 0020 trap: rows kill kernels).

## Leakage rule (the one that makes a fake win)

Every stacked value on row *t* must come from an SVM fit on rows
**strictly before** *t* (walk-forward OOF). Fitting once on the whole
pooled frame and appending predictions leaks the label into the feature.
That is the single easiest way to manufacture a win here.

Two clocks — do not mix them:

- **Nightly (fit):** the three estimators train on all history once and
  are persisted next to the logistic diagnostic. LinearSVR dominates
  (~35–45 min); the two LinearSVCs are smaller. This is the 3:30 CT
  retrain, not the morning scan.
- **Morning (predict):** today's open-known row is already built
  (`live_rows.prepare_live_rows`, the current <2 min wall). Scoring the
  three stacked columns is three linear predicts on that same ~90-feature
  vector — a matrix-vector multiply per name, milliseconds across the
  universe, not a refit. The morning scan does **not** run
  `svr_stack_search`.

State the nightly fit cost in the promotion PR; do not discover it after.

## Promotion bar

Same empirical gate as everything else (ADR 0004, 0008):

1. Research script, walk-forward folds only, holdout untouched. Solo
   LightGBM on the existing feature set is always the baseline in the
   same run.
2. LightGBM-with-the-three-columns must beat that baseline in **≥3 of 4
   folds on both acc and rank IC**. Per-fold lists, not just means
   (0020's lesson).
3. If (2) holds, score holdout once on the 200 never-seen tickers.
   Both the folds and the holdout must favor the stacked model.
4. Either failure → keep the script, flip this record to Rejected.
   Success → add the columns on the training-frame path, persist the
   three estimators, flip this record to Accepted with the holdout
   numbers.

Do not mix 0020's blend table with this search. Different question,
same snapshot discipline: rerun the baseline in the same job.

## Work queue

1. `python/stock_picker/training/svr_stack_search.py` — exists; run
   `bazelisk run //python/stock_picker/training:svr_stack_search`.
   Walk-forward, holdout untouched. Each fold fits LinearSVR + two
   LinearSVCs on train, attaches the three columns, fits LightGBM with
   and without them. Test-side columns are OOF; train-side columns are
   in-sample so LightGBM has a column to split on (standard stacking
   shortcut — scored metrics are the OOF test ones). Background job,
   expect the LinearSVR fold times from 0020 (7–43 min/fold).
2. Read the per-fold lists against the bar above.
3. Holdout once, or close it out.

Optional, each its own script, only after 1–3 settles:

- Kernel PCA swap for `features/structure.py`'s linear PCA (per-day
  cross-section ~1,800 names, so O(n²) is fine). This is a real
  feature-store change and needs ADR 0007's three views.
- RankSVM (Joachims 2002) as an `objective_search.py` candidate for
  Rank, sampled pairs, not all-pairs.

## Traps

- **OOF or don't bother.** In-sample SVM columns will look great and
  be worthless live.
- **These columns are open-known at score time** only in the sense that
  the live row already is — they are functions of the same open-known
  feature vector Fit already scores. They are not close-known market
  features. Do not shift(1) them as if they were yesterday's PCA.
- **Don't put them in `PREDICTIVE_MODEL_TYPES`.** That would revive the
  blender path 0020 just closed.
- LinearSVR ConvergenceWarnings and all-NaN early-window imputer
  warnings are known (0020). max_iter stays 10000.
- gazelle is still broken; hand-edit BUILD.bazel for the new script.

## Why this can work after the blend failed

Blending asks SVR to be a second forecast whose errors cancel
LightGBM's when averaged. Stacking asks LightGBM to *use* SVR's number
when it wants and ignore it when it doesn't — a split the trees can
learn per regime, which a fixed (3,1,0) weight cannot. The overnight
rerun said the fixed weight was unstable. It did not say the linear
view contains no information.
