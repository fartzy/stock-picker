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
from stock_picker.training.model import evaluate, feature_columns
from stock_picker.training.splits import select_holdout_tickers
from stock_picker.training.train import run_walk_forward

MODEL_NAME = "day_session_return"


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

    train_dataset = _load_pooled_dataset(train_tickers, price_store, feature_store)

    fold_results = run_walk_forward(train_dataset, excluded_features=excluded_features)
    for result in fold_results:
        print(f"fold {result['fold']}: {result['metrics']}")

    final_model = fold_results[-1]["model"]
    ModelStore().write(MODEL_NAME, final_model)

    if not holdout_tickers:
        return

    holdout_dataset = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
    holdout_metrics = evaluate(final_model, holdout_dataset, excluded_features=excluded_features)
    print(f"holdout tickers {holdout_tickers}: {holdout_metrics}")

    columns = feature_columns(holdout_dataset, excluded_features=excluded_features)
    predicted = pd.Series(
        final_model.predict(holdout_dataset[columns]), index=holdout_dataset.index
    )
    actual = holdout_dataset[LABEL_COLUMN]
    sweep = sweep_thresholds(predicted, actual)
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
