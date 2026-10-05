# 0023: Keep all seven SVM outputs in the Fit feature landscape

Date: 2026-10-04
Status: Implemented

ADR 0021 found no consistently winning seven-output configuration. ADR 0022
nevertheless added `svc_direction_margin` as a removable LightGBM input. The
owner now chooses to include the other six as candidates too, without treating
the old promotion gate as a prerequisite. This is an inclusion decision, not a
claim of measured improvement.

All seven outputs in `features/stacked_svm.py` are available to Fit LightGBM
unless selected out or pruned. They are model-derived, not stored in market
feature parquet. Each training-row value comes from a LinearSVR or LinearSVC
fitted on strictly earlier dates; the first warm-up block is excluded. The
last fold saves its paired estimators beside LightGBM, and morning inference
scores the live open-known row with those saved estimators. Rank and other Fit
model families do not consume the outputs.

Pruning or positive selection changes the **next** trained model. An archived
model keeps its own feature list and estimators, so selecting an older run
does not silently change its predictions. Training only replaces `latest`
after the full evaluation succeeds. The LinearSVR makes all-seven retraining
materially slower than the one-margin path; watch actual run time before
relying on a completed nightly model at the next open.
