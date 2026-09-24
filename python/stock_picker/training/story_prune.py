"""Rank the new open-known story columns, then drop the ones that lose.

Uses three lenses on the same walk-forward train rows (holdout tickers out):

- RandomForest impurity (nonlinear, interactions)
- Ridge |coef_| after StandardScaler (linear effect size, all features kept)
- Lasso |coef_| after StandardScaler (same scale; zeros are a drop vote)

A story column is pruned only if it is below-median on RF *and* below-median
on Ridge, or Lasso zeros it and RF also ranks it in the bottom half.
Does not touch production LightGBM until you apply the prune list.

Run: bazelisk run //python/stock_picker/training:story_prune
     bazelisk run //python/stock_picker/training:story_prune -- --apply
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_picker.features.open_pattern_seasonality import STORY_OPEN_KNOWN_COLUMNS
from stock_picker.features.pruning import pruned_features
from stock_picker.log import get_logger
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.model import (
    RIDGE_DEFAULT_PARAMS,
    feature_columns,
    train_random_forest,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

logger = get_logger(__name__)

NAMED_EXPERIMENT_COLUMNS = (
    "two_big_then_fade_open3_seasonality",
    "month_crash_yday_up_open3_seasonality",
    "week_run_open3_seasonality",
    "chase_then_fade_open3_seasonality",
    "three_up_open3_seasonality",
    "dump_then_quiet_open3_seasonality",
)
EXPERIMENT_COLUMNS = tuple(NAMED_EXPERIMENT_COLUMNS) + tuple(STORY_OPEN_KNOWN_COLUMNS)
LASSO_ALPHA = 1e-4
PRUNE_REASON = "story prune: below-median RF and Ridge (or Lasso zero)"


def experiment_columns_present(columns: list[str]) -> list[str]:
    return [name for name in EXPERIMENT_COLUMNS if name in columns]


def _load_pooled(tickers, price_store, feature_store):
    histories, features_by_ticker = {}, {}
    for ticker in tickers:
        try:
            histories[ticker] = price_store.read(ticker)
            features_by_ticker[ticker] = feature_store.read(ticker)
        except FileNotFoundError:
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def _linear_abs_coefs(estimator, columns: list[str]) -> dict[str, float]:
    coefs = estimator.named_steps["regress"].coef_
    return {name: float(abs(value)) for name, value in zip(columns, coefs)}


def _fit_standardized_linear(model, train_frame: pd.DataFrame, columns: list[str]):
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("regress", model),
        ]
    )
    pipe.fit(train_frame[columns], train_frame[LABEL_COLUMN])
    return pipe


def rank_experiment_columns(train_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per experiment column that is actually in `columns`."""
    present = experiment_columns_present(columns)
    forest = train_random_forest(train_frame, included_features=set(columns))
    rf = {
        name: float(gain)
        for name, gain in zip(forest.feature_names, forest.estimator.feature_importances_)
    }
    ridge = _fit_standardized_linear(
        Ridge(**RIDGE_DEFAULT_PARAMS), train_frame, columns
    )
    lasso = _fit_standardized_linear(
        Lasso(alpha=LASSO_ALPHA, random_state=0, max_iter=4000), train_frame, columns
    )
    ridge_abs = _linear_abs_coefs(ridge, columns)
    lasso_abs = _linear_abs_coefs(lasso, columns)

    rows = []
    for name in present:
        rows.append(
            {
                "feature": name,
                "rf": rf.get(name, 0.0),
                "ridge_abs": ridge_abs.get(name, 0.0),
                "lasso_abs": lasso_abs.get(name, 0.0),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        table["drop"] = []
        return table
    table["rf_rank"] = table["rf"].rank(ascending=False, method="min")
    table["ridge_rank"] = table["ridge_abs"].rank(ascending=False, method="min")
    table["lasso_zero"] = table["lasso_abs"] <= 1e-12
    weak_rf = table["rf"] <= table["rf"].quantile(0.5)
    weak_ridge = table["ridge_abs"] <= table["ridge_abs"].quantile(0.5)
    table["drop"] = (weak_rf & weak_ridge) | (table["lasso_zero"] & weak_rf)
    return table.sort_values(["drop", "rf"], ascending=[True, False]).reset_index(drop=True)


def columns_to_prune(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return []
    return table.loc[table["drop"], "feature"].tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write drops to PrunedFeatureStore. Default is print-only.",
    )
    args = parser.parse_args(argv)

    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train_tickers = [ticker for ticker in tickers if ticker not in holdout]
    logger.info("loading %s train tickers (%s holdout)", len(train_tickers), len(holdout))
    pooled = _load_pooled(train_tickers, PriceStore(), FeatureStore())
    excluded = pruned_features()
    columns = feature_columns(pooled, excluded)
    present = experiment_columns_present(columns)
    logger.info("rows=%s features=%s experiment=%s", len(pooled), len(columns), len(present))
    if not present:
        logger.warning("no experiment columns in the pooled frame -- rebuild features first")
        return 1

    splits = walk_forward_splits(pooled["date"], n_splits=4)
    train_mask, _test_mask = splits[-1]
    train_frame = pooled[train_mask]
    table = rank_experiment_columns(train_frame, columns)
    logger.info("\n%s", table.to_string(index=False))
    drops = columns_to_prune(table)
    logger.info("drop %s: %s", len(drops), ", ".join(drops) or "(none)")
    if args.apply and drops:
        store = PrunedFeatureStore()
        for name in drops:
            store.prune(name, reason=PRUNE_REASON)
        logger.info("wrote %s names to PrunedFeatureStore", len(drops))
    elif not args.apply:
        logger.info("print-only; pass --apply to exclude drops from the next train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
