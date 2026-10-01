# 0022: Include the direction SVC margin as a Fit model-derived feature

Date: 2026-09-30
Status: Accepted (implementation pending; no new model selected for live use)

## Context and evidence

ADR 0021 tested seven SVM outputs, individually and in subsets, as extra
LightGBM training columns. That research used strictly earlier-date SVM fits
to generate both training and validation values. On the same eligible rows,
solo LightGBM averaged 50.327% directional accuracy and 0.06912 rank IC;
adding only `svc_direction_margin` averaged 50.518% and 0.07217. The margin
improved accuracy in 2 of 4 walk-forward folds and rank IC in 3 of 4. It
therefore failed ADR 0021's predeclared requirement of at least 3 of 4 wins
on **both** metrics. The 200 held-out tickers were not scored. These small
research-fold selection gains do not establish a repeatable or holdout
benefit. The full per-fold results and immutable run manifest are in
`stock-picker-research/svm-derived-features/2026-09-30-full-bd7ede1f/checkpoints/`
beside this repository.

The product owner chooses to keep a potentially useful linear
summary in the Fit feature landscape, then assess and prune it from future
archived training runs if it proves unhelpful. A feature may contribute a
small amount without qualifying as a standalone forecast or blender member.

## Decision

- Include only `svc_direction_margin` by default in **future Fit/LightGBM
  training runs**, as a model-derived candidate feature that can be turned
  off or pruned. Do not include the other six ADR 0021 outputs by default.
  Do not add an SVC to `PREDICTIVE_MODEL_TYPES`, a Fit ensemble blend, or the
  parallel Rank model. ADR 0008's empirical ensemble-membership rule is
  unchanged.
- Set aside ADR 0021's 3-of-4-fold and holdout *promotion gate for this one
  feature*. Keep the failed-gate result and unscored holdout visible; do not
  describe this choice as evidence that the feature generalizes. ADR 0004's
  chronological validation and ticker-holdout design remain in force for
  evaluating later trained runs.
- Generate every training-row margin with a direction SVC fitted on strictly
  earlier dates (out-of-fold), using the same open-known feature vector and
  `return > 0` target tested in ADR 0021. Never fit the SVC on a row and
  feed its prediction for that row to LightGBM. Exclude early warm-up rows
  lacking any valid earlier fit, with comparisons on the same eligible rows.
  For validation and holdout, fit only on their permitted prior/training rows.
  Preserve ADR 0003's open-to-close label and close-known/open-known timing
  rules.
- Persist the final direction SVC and its preprocessing, feature order, and
  metadata with each new Fit model that uses this column. At inference,
  compute its margin from the live open-known row before LightGBM scores.
  Fail visibly when a new model expects the margin but its estimator or
  required inputs are missing; never silently substitute `NaN`. Existing
  archived models must continue to score without an SVC.
- Expose the margin as a **model-derived** feature in training selection,
  diagnostics, and pruning. It is not a market-data parquet column or a
  pandas formula in `features/pipeline.py`; do not fabricate ADR 0007's
  market-feature formula/example views for it. Document its estimator,
  target, source inputs, timing, and trained-model dependency where users
  inspect or select model-derived features.
- Archive new runs normally. If `selected_run_id` pins a prior archive,
  live scoring stays on that archive until the user changes the selection,
  per ADR 0013. When no run is pinned, live scoring follows `latest`, so a
  successful retraining updates it automatically. This decision does not
  itself retrain, select, or promote a live model.

## Consequences and verification

The nightly Fit run gains one linear classifier fit plus chronological
out-of-fold generation; the morning scan only evaluates its saved decision
function. Measure the actual runtime in implementation rather than assuming
the seven-estimator research run's cost applies to this one-column path.
The feature is deliberately reversible through the training selection and
the archived-run picker.

Implementation must test strict date ordering, no held-out ticker in SVC
training, finite margin values, pruning/selection behavior, serialization
and inference parity, legacy-archive compatibility, and a missing-estimator
failure. Compare subsequent archived runs against the unmodified baseline
and report per-fold and holdout results where available; do not claim the
unrun holdout as a validation result.
