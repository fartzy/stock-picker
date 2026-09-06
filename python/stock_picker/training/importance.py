"""Per-model-type feature importance, aggregated across an ensemble -- a
stronger "does this feature matter" signal than coverage alone, surfaced in
the Feature Store UI alongside coverage/correlation to inform pruning
decisions.
"""

from __future__ import annotations

from stock_picker.storage.model_store import ModelStore
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.main import MODEL_NAME
from stock_picker.training.model import TrainedModel


def model_type_importance(trained: TrainedModel) -> dict[str, float]:
    """Percent of total importance per feature (sums to ~100), normalized the
    same way regardless of which library actually produced the raw numbers."""
    if trained.model_type == "lightgbm":
        names = trained.estimator.feature_name()
        gains = list(trained.estimator.feature_importance(importance_type="gain"))
    elif trained.model_type == "random_forest":
        names = trained.feature_names
        gains = list(trained.estimator.feature_importances_)
    elif trained.model_type == "logistic_regression":
        # trained.estimator is a Pipeline (impute -> classify), see
        # model.py's train_logistic_regression for why.
        names = trained.feature_names
        classifier = trained.estimator.named_steps["classify"]
        gains = [abs(coefficient) for coefficient in classifier.coef_[0]]
    else:
        raise ValueError(f"unknown model_type: {trained.model_type!r}")

    total = float(sum(gains))
    if total == 0:
        return {name: 0.0 for name in names}
    return {name: float(gain) / total * 100 for name, gain in zip(names, gains)}


def ensemble_importance(ensemble: Ensemble) -> dict[str, float]:
    """Weighted average of each member's own importance. Members can have
    different feature subsets (e.g. one trained only on momentum features) --
    a feature a given member never saw simply contributes 0 from it, same as
    if that member had assigned it zero importance."""
    total_weight = sum(ensemble.weights)
    combined: dict[str, float] = {}
    for member, weight in zip(ensemble.members, ensemble.weights):
        for name, value in model_type_importance(member).items():
            combined[name] = combined.get(name, 0.0) + value * weight / total_weight
    return combined


def model_importance() -> dict:
    """Wiring: loads the persisted production ensemble and returns both its
    blended importance and a per-model-type breakdown, or empty dicts if no
    model has been trained yet. The breakdown matters because a weight=0
    diagnostic member (e.g. logistic_regression, see ensemble.py's default
    spec) always contributes zero to the blend by construction -- its own
    importance is only visible here, not in `blended`."""
    store = ModelStore()
    if not store.exists(MODEL_NAME):
        return {"blended": {}, "by_model_type": {}}
    ensemble = store.read(MODEL_NAME)
    return {
        "blended": ensemble_importance(ensemble),
        "by_model_type": {
            member.model_type: model_type_importance(member) for member in ensemble.members
        },
    }
