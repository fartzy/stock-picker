"""Fit and persist the ListFold rank model next to production LightGBM.

Does not overwrite day_session_return.pkl. Learning-to-rank (ListFold)
only penalizes wrong *order* within a day -- not |actual% − predicted%|.
A winner scored "too low" in percent terms is fine if it still ranks above
losers. Scores are relative, not percents; morning serving takes top-K,
not a 0.5% gate. See train_lightgbm_rank in model.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from stock_picker.features.pruning import PRICE_PATTERN_COLUMNS, pruned_features
from stock_picker.log import get_logger
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble, ModelSpec, train_ensemble
from stock_picker.training.main import _load_pooled_dataset
from stock_picker.training.splits import select_holdout_tickers

RANK_MODEL_NAME = "day_session_return_rank"
RANK_TOP_K = 20
# News / sentiment only on this many Rank names so 8:36 isn't waiting on 20 Finnhub calls.
RANK_NEWS_TOP_K = 10

logger = get_logger(__name__)


def activate_rank_artifact(path: Path, *, store: ModelStore | None = None) -> dict:
    """Activate a validated fitted Rank ensemble, keeping a recoverable backup."""
    source = ModelStore(data_dir=path.parent).read(path.stem)
    if not isinstance(source, Ensemble) or not source.members or any(
        member.model_type != "lightgbm_rank" or not callable(getattr(member.estimator, "predict", None))
        for member in source.members
    ) or sum(source.weights) <= 0:
        raise ValueError("activation requires a fitted Rank ensemble")
    names = {name for member in source.members for name in member.feature_names}
    if not names:
        raise ValueError("Rank artifact has no feature schema")
    excluded = names & pruned_features()
    if excluded:
        raise ValueError(f"Rank artifact contains currently pruned features: {sorted(excluded)}")
    destination = store if store is not None else ModelStore()
    backup = None
    if destination.exists(RANK_MODEL_NAME):
        backup = f"{RANK_MODEL_NAME}_before_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        ModelStore(data_dir=destination.data_dir / "backups").write(backup, destination.read(RANK_MODEL_NAME))
    destination.write(RANK_MODEL_NAME, source)
    return {
        "active_model": RANK_MODEL_NAME, "source": str(path),
        "feature_count": len(names), "new_feature_count": len(names & PRICE_PATTERN_COLUMNS),
        "backup": backup,
    }


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate-artifact", type=Path)
    args = parser.parse_args()
    if args.activate_artifact:
        logger.info("Rank activated: %s", activate_rank_artifact(args.activate_artifact))
        return
    train_and_persist_rank_model()
    logger.info("wrote %s.pkl", RANK_MODEL_NAME)


if __name__ == "__main__":
    main()
