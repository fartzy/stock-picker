"""Entrypoint: train the day-session return model over all actively tracked tickers."""

from __future__ import annotations

from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.dataset import build_pooled_dataset
from stock_picker.training.train import run_walk_forward

MODEL_NAME = "day_session_return"


def main() -> None:
    tickers = UniverseStore().active_tickers()

    price_store = PriceStore()
    feature_store = FeatureStore()
    histories = {ticker: price_store.read(ticker) for ticker in tickers}
    features_by_ticker = {ticker: feature_store.read(ticker) for ticker in tickers}

    pooled_dataset = build_pooled_dataset(histories, features_by_ticker)

    fold_results = run_walk_forward(pooled_dataset)
    for result in fold_results:
        print(f"fold {result['fold']}: {result['metrics']}")

    final_model = fold_results[-1]["model"]
    ModelStore().write(MODEL_NAME, final_model)


if __name__ == "__main__":
    main()
