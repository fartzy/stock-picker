"""Entrypoint: print the feature catalog and a coverage report for the current
universe. `bazel run //python/stock_picker/features:catalog` is the answer to
"how do I see the features."
"""

from __future__ import annotations

from stock_picker.features.catalog import coverage_report, describe_all, list_feature_columns
from stock_picker.features.catalog_loader import active_tickers, feature_tables, sample_history
from stock_picker.log import get_logger

logger = get_logger(__name__)


def main() -> None:
    tickers = active_tickers()
    history = sample_history(tickers)

    catalog = list_feature_columns(history)
    descriptions = describe_all(history)
    total = sum(len(columns) for columns in catalog.values())
    logger.info("Feature catalog: %s columns across %s categories", total, len(catalog))
    for category, columns in catalog.items():
        logger.info("[%s] (%s)", category, len(columns))
        for column in columns:
            logger.info("  %s: %s", column, descriptions.get(column, ""))

    report = coverage_report(feature_tables(tickers))
    logger.info("Coverage report across %s tickers (lowest first):", len(tickers))
    logger.info("%s", report.to_string())


if __name__ == "__main__":
    main()
