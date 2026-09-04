"""Entrypoint: print the feature catalog and a coverage report for the current
universe. `bazel run //python/stock_picker/features:catalog` is the answer to
"how do I see the features."
"""

from __future__ import annotations

from stock_picker.features.catalog import coverage_report, describe_all, list_feature_columns
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore


def main() -> None:
    tickers = UniverseStore().active_tickers()

    price_store = PriceStore()
    sample_history = price_store.read(tickers[0])

    catalog = list_feature_columns(sample_history)
    descriptions = describe_all(sample_history)
    total = sum(len(columns) for columns in catalog.values())
    print(f"Feature catalog: {total} columns across {len(catalog)} categories\n")
    for category, columns in catalog.items():
        print(f"[{category}] ({len(columns)})")
        for column in columns:
            print(f"  {column}: {descriptions.get(column, '')}")
        print()

    feature_store = FeatureStore()
    feature_tables = {ticker: feature_store.read(ticker) for ticker in tickers}
    report = coverage_report(feature_tables)
    print(f"Coverage report across {len(tickers)} tickers (lowest first):")
    print(report.to_string())


if __name__ == "__main__":
    main()
