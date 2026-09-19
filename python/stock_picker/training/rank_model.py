"""Fit and persist the lambdarank model next to production LightGBM.

Does not overwrite day_session_return.pkl. Learning-to-rank (lambdarank)
only penalizes wrong *order* within a day -- not |actual% − predicted%|.
A winner scored "too low" in percent terms is fine if it still ranks above
losers. Scores are relative, not percents; morning serving takes top-K,
not a 0.5% gate. See train_lightgbm_rank in model.py.
"""

from __future__ import annotations

from stock_picker.features.pruning import pruned_features
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble, ModelSpec, train_ensemble
from stock_picker.training.main import _load_pooled_dataset
from stock_picker.training.splits import select_holdout_tickers

RANK_MODEL_NAME = "day_session_return_rank"
RANK_TOP_K = 20


def train_and_persist_rank_model(
    model_store: ModelStore | None = None,
    included_features: set[str] | None = None,
) -> Ensemble:
    excluded = pruned_features()
    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout]
    pooled = _load_pooled_dataset(train_tickers, PriceStore(), FeatureStore())
    specs = [
        ModelSpec(
            "lightgbm_rank",
            excluded_features=excluded,
            included_features=included_features,
            weight=1.0,
        )
    ]
    ensemble = train_ensemble(pooled, specs)
    store = model_store or ModelStore()
    store.write(RANK_MODEL_NAME, ensemble)
    return ensemble


def main() -> None:
    train_and_persist_rank_model()
    print(f"wrote {RANK_MODEL_NAME}.pkl", flush=True)


if __name__ == "__main__":
    main()
