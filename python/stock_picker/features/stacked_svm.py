"""Metadata for model-derived SVM stacking outputs (ADRs 0021 and 0022).

These values are produced by fitted training estimators, not by the feature
pipeline or its persisted parquet snapshots. Only the direction margin is
eligible for production LightGBM training; the other six remain research-only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentalFeature:
    name: str
    description: str
    computation: str
    example: str


SVM_DERIVED_FEATURES = (
    ExperimentalFeature(
        name="svr_oof_pred",
        description="LinearSVR's forecast of the stock's open-to-close return.",
        computation="Fit LinearSVR on earlier labeled days; predict on this day's open-known features. Training rows require walk-forward out-of-fold predictions.",
        example="A value of 0.006 predicts a +0.6% open-to-close return; it is not an observed return.",
    ),
    ExperimentalFeature(
        name="svc_direction_margin",
        description="Signed LinearSVC margin for a positive open-to-close return.",
        computation="Fit LinearSVC out-of-fold on earlier days using all unpruned raw open-known inputs and return > 0 as the positive class; score live rows with the saved estimator's decision_function.",
        example="A positive margin favors an up day; its magnitude is a model score, not a probability.",
    ),
    ExperimentalFeature(
        name="svc_gate_margin",
        description="Signed LinearSVC margin for the +0.5% return gate used by the buy signal.",
        computation="Fit LinearSVC on earlier days with return > +0.5% as the positive class; use its decision_function margin.",
        example="A positive margin favors clearing +0.5%; a margin of 1 does not mean a 1% return.",
    ),
    ExperimentalFeature(
        name="svc_down_gate_margin",
        description="Signed LinearSVC margin for the −0.5% downside-return boundary.",
        computation="Fit LinearSVC on earlier days with return < −0.5% as the positive class; use its decision_function margin.",
        example="A positive margin favors a decline worse than −0.5%; it is not a calibrated probability.",
    ),
    ExperimentalFeature(
        name="svc_strong_up_margin",
        description="Signed LinearSVC margin for the +1% strong-up return boundary.",
        computation="Fit LinearSVC on earlier days with return > +1% as the positive class; use its decision_function margin.",
        example="A positive margin favors an open-to-close gain above +1%.",
    ),
    ExperimentalFeature(
        name="svc_top_quintile_margin",
        description="Signed score for being among the day's top 20% open-to-close returns.",
        computation="On earlier days, label each day's top return quintile, fit LinearSVC, then score this day's open-known features.",
        example="A higher margin favors a top-quintile rank; it is not the predicted return.",
    ),
    ExperimentalFeature(
        name="svc_bottom_quintile_margin",
        description="Signed score for being among the day's bottom 20% open-to-close returns.",
        computation="On earlier days, label each day's bottom return quintile, fit LinearSVC, then score this day's open-known features.",
        example="A higher margin favors a bottom-quintile rank; it is not the predicted return.",
    ),
)

STACKED_SVM_COLUMNS = tuple(feature.name for feature in SVM_DERIVED_FEATURES)
PRODUCTION_MODEL_DERIVED_COLUMNS = ("svc_direction_margin",)
RESEARCH_SVM_COLUMNS = tuple(
    name for name in STACKED_SVM_COLUMNS if name not in PRODUCTION_MODEL_DERIVED_COLUMNS
)
