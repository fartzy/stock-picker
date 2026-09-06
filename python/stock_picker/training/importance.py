"""LightGBM gain-based feature importance -- a stronger "does this feature
matter" signal than coverage alone, surfaced in the Feature Store UI alongside
coverage/correlation to inform pruning decisions.
"""

from __future__ import annotations

import lightgbm as lgb

from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.paths import data_root
from stock_picker.training.main import MODEL_NAME


def gain_importance(model: lgb.Booster) -> dict[str, float]:
    """Feature name -> percent of total gain (sums to ~100), so it reads as
    "how much this matters" rather than a raw unitless LightGBM gain number."""
    names = model.feature_name()
    gains = model.feature_importance(importance_type="gain")
    total = float(sum(gains))
    if total == 0:
        return {name: 0.0 for name in names}
    return {name: float(gain) / total * 100 for name, gain in zip(names, gains)}


def model_importance() -> dict[str, float]:
    """Wiring: loads the persisted production model and returns its gain
    importance, or an empty dict if no model has been trained yet."""
    model_path = data_root() / "models" / f"{MODEL_NAME}.txt"
    if not model_path.exists():
        return {}
    return gain_importance(ModelStore().read(MODEL_NAME))
