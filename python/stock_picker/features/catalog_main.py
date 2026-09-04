"""Entrypoint: print the feature catalog and a coverage report for the current
universe. `bazel run //python/stock_picker/features:catalog` is the answer to
"how do I see the features."
"""

from __future__ import annotations

from stock_picker.features.catalog import coverage_report, describe_all, list_feature_columns
from stock_picker.features.catalog_loader import active_tickers, feature_tables, sample_history


def main() -> None:
    tickers = active_tickers()
    history = sample_history(tickers)

    catalog = list_feature_columns(history)
    descriptions = describe_all(history)
    total = sum(len(columns) for columns in catalog.values())
    print(f"Feature catalog: {total} columns across {len(catalog)} categories\n")
    for category, columns in catalog.items():
        print(f"[{category}] ({len(columns)})")
        for column in columns:
            print(f"  {column}: {descriptions.get(column, '')}")
        print()

    report = coverage_report(feature_tables(tickers))
    print(f"Coverage report across {len(tickers)} tickers (lowest first):")
    print(report.to_string())


if __name__ == "__main__":
    main()
