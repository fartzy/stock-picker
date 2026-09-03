"""Entrypoint: train the day-session return model, holding out a few tickers entirely
to test whether the model's signal generalizes to stocks it has never seen -- not just
future dates for tickers it has already seen (that's what walk-forward validates).
"""

from __future__ import annotations

import pandas as pd

from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import sweep_thresholds
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.model import evaluate, feature_columns
from stock_picker.training.train import run_walk_forward

MODEL_NAME = "day_session_return"

# Held out of training entirely -- tests whether the pooled model's signal
# generalizes to stocks it has never seen, not just future dates for tickers it has.
HOLDOUT_TICKERS = {"JNJ", "XOM", "WMT"}


def _load_pooled_dataset(
    tickers: list[str], price_store: PriceStore, feature_store: FeatureStore
) -> pd.DataFrame:
    histories = {ticker: price_store.read(ticker) for ticker in tickers}
    features_by_ticker = {ticker: feature_store.read(ticker) for ticker in tickers}
    return build_pooled_dataset(histories, features_by_ticker)


def main() -> None:
    tickers = UniverseStore().active_tickers()
    train_tickers = [t for t in tickers if t not in HOLDOUT_TICKERS]
    holdout_tickers = [t for t in tickers if t in HOLDOUT_TICKERS]

    price_store = PriceStore()
    feature_store = FeatureStore()

    train_dataset = _load_pooled_dataset(train_tickers, price_store, feature_store)

    fold_results = run_walk_forward(train_dataset)
    for result in fold_results:
        print(f"fold {result['fold']}: {result['metrics']}")

    final_model = fold_results[-1]["model"]
    ModelStore().write(MODEL_NAME, final_model)

    if not holdout_tickers:
        return

    holdout_dataset = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
    holdout_metrics = evaluate(final_model, holdout_dataset)
    print(f"holdout tickers {holdout_tickers}: {holdout_metrics}")

    columns = feature_columns(holdout_dataset)
    predicted = pd.Series(
        final_model.predict(holdout_dataset[columns]), index=holdout_dataset.index
    )
    actual = holdout_dataset[LABEL_COLUMN]
    sweep = sweep_thresholds(predicted, actual)
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
