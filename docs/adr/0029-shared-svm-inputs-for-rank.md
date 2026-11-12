# 0029: Share the seven SVM inputs with Rank

Date: 2026-11-12
Status: Implemented for subsequent training; current live artifacts unchanged.

Rank can now select the same seven model-derived inputs as Fit. This supersedes
ADR 0023's Fit-only restriction. Each input follows normal positive selection
and pruning; adding it is not evidence of better returns.

The normal training run generates chronological out-of-fold SVM outputs once.
Fit and the separate Rank head consume that shared frame and share the final
fitted SVM estimators. Each training value comes from strictly earlier dates;
warm-up rows without such a value are excluded from a head that uses SVM inputs.
A raw-only head retains the full raw training set. The SVMs use the same
unpruned raw source schema, irrespective of a head's narrower positive selection.
Conflicting raw exclusions across stacked heads are rejected.

Rank stays outside the return ensemble. Its scores are relative rankings, not
return percentages. The final Rank model is saved separately, including a
run-specific archive when a run ID is supplied. Each archive owns its feature
schema, preprocessing, and paired SVMs; inference recomputes the inputs using
those estimators and ignores caller-supplied SVM values. Old raw Rank archives
still work. The run record's `resolved_features` continues to describe Fit;
Rank's own archive describes its inputs.

The standalone Rank command uses the same chronological training path. Without
an in-memory shared Fit run it must build its own out-of-fold values; it cannot
substitute predictions from today's saved full-history SVMs on training rows.
Joint training adds one Rank head fit, not another seven-SVM training pass.
Standalone Rank folds report Rank IC and top-1/5/20 returns, never return MAE
or sign accuracy on its unitless scores. Joint-run fold metrics still describe Fit.

Tests cover shared fit counts and OOF frames, pruning/selection, raw-only row
coverage, held-out ticker isolation, archive round trips, missing estimators,
legacy Rank inference, and publication only after required fits/evaluations.
This change does not run production training or establish an accuracy gain.
