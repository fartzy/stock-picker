"""Entrypoint: train the day-session return model, holding out a few tickers entirely
to test whether the model's signal generalizes to stocks it has never seen -- not just
future dates for tickers it has already seen (that's what walk-forward validates).
"""

from __future__ import annotations

import pandas as pd

from stock_picker.features.pruning import pruned_features
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import sweep_thresholds
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.ensemble import ModelSpec, evaluate_ensemble, predict_ensemble
from stock_picker.training.splits import select_holdout_tickers
from stock_picker.training.train import run_walk_forward

MODEL_NAME = "day_session_return"
# The default ensemble composition -- a developer edits this directly to
# experiment (add a model type, change a weight). `excluded_features` (the
# pruned set) is applied to each member when built in main(), since it's only
# known at runtime.
DEFAULT_MODEL_TYPES = ["lightgbm", "random_forest"]
# Diagnostic-only: weight=0.0 means it never affects the blended prediction
# (ensemble.predict_ensemble's weighted average) or the blended importance --
# it exists purely so its own coefficient-based importance is visible via
# importance.model_importance()'s by_model_type breakdown, a genuinely
# different lens (linear/monotonic effect size) than the tree-based
# gain/impurity measures the other two model types produce.
DIAGNOSTIC_MODEL_TYPES = ["logistic_regression"]


def _load_pooled_dataset(
    tickers: list[str], price_store: PriceStore, feature_store: FeatureStore
) -> pd.DataFrame:
    histories = {}
    features_by_ticker = {}
    for ticker in tickers:
        try:
            histories[ticker] = price_store.read(ticker)
            features_by_ticker[ticker] = feature_store.read(ticker)
        except FileNotFoundError:
            # Active in UniverseStore doesn't guarantee price/feature data exists
            # for it (e.g. a transient ingestion failure) -- skip rather than
            # crash the whole run over one ticker.
            print(f"skipping {ticker}: missing price or feature data")
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def main() -> None:
    tickers = UniverseStore().active_tickers()
    holdout_ticker_set = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout_ticker_set]
    holdout_tickers = [t for t in tickers if t in holdout_ticker_set]

    price_store = PriceStore()
    feature_store = FeatureStore()
    excluded_features = pruned_features()
    specs = [
        ModelSpec(model_type, excluded_features=excluded_features) for model_type in DEFAULT_MODEL_TYPES
    ] + [
        ModelSpec(model_type, excluded_features=excluded_features, weight=0.0)
        for model_type in DIAGNOSTIC_MODEL_TYPES
    ]

    train_dataset = _load_pooled_dataset(train_tickers, price_store, feature_store)

    fold_results = run_walk_forward(train_dataset, specs=specs)
    for result in fold_results:
        print(f"fold {result['fold']}: {result['metrics']}")

    final_ensemble = fold_results[-1]["model"]
    ModelStore().write(MODEL_NAME, final_ensemble)

    if not holdout_tickers:
        return

    holdout_dataset = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
    holdout_metrics = evaluate_ensemble(final_ensemble, holdout_dataset)
    print(f"holdout tickers {holdout_tickers}: {holdout_metrics}")

    predicted = pd.Series(predict_ensemble(final_ensemble, holdout_dataset), index=holdout_dataset.index)
    actual = holdout_dataset[LABEL_COLUMN]
    sweep = sweep_thresholds(predicted, actual)
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
