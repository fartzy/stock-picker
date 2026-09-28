"""Walk-forward bake-off of LightGBM with vs without stacked SVM columns.

ADR 0021: LinearSVR is closed as a blender member (ADR 0020). This asks
whether seven out-of-fold SVM numbers help LightGBM as extra columns:

  svr_oof_pred                 LinearSVR predicted open->close return
  svc_direction_margin         LinearSVC distance to the up/down plane
  svc_gate_margin              LinearSVC distance to the +0.5% buy-gate
  svc_down_gate_margin         LinearSVC distance to the -0.5% tail
  svc_strong_up_margin         LinearSVC distance to the +1% plane
  svc_top_quintile_margin      LinearSVC distance to within-day top 20%
  svc_bottom_quintile_margin   LinearSVC distance to within-day bottom 20%

One LinearSVR (the expensive nightly fit) plus six LinearSVC planes that
each ask a different question. Extra SVRs are not in this search -- each
would add ~35-45 min nightly for a correlated return forecast.

Each fold fits the seven estimators on TRAIN only, attaches their
outputs to the TEST frame, then fits LightGBM with and without those
columns. Holdout tickers are never touched. Does not write production
pickles or touch PrunedFeatureStore.

Promotion bar (ADR 0021): LightGBM-with-the-stacked-columns beats solo
LightGBM in >=3 of 4 folds on both acc and rank IC. Per-fold lists, not
just means. Holdout is a later, separate step if this clears.

Run: bazelisk run //python/stock_picker/training:svr_stack_search
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stock_picker.features.pruning import pruned_features
from stock_picker.log import get_logger
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import rank_ic, simulate_trades
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.model import (
    STACKED_SVM_COLUMNS,
    attach_stacked_svm_columns,
    fit_stacked_svm_estimators,
    predict,
    train_lightgbm,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

logger = get_logger(__name__)

N_SPLITS = 4
FIT_THRESHOLD = 0.005


def _load_pooled(tickers, price_store, feature_store):
    # Do not import training.main -- that module is both a py_library source
    # and a py_binary, and gazelle cannot resolve the import (same pattern
    # as tune_experiment.py).
    histories, features_by_ticker = {}, {}
    for ticker in tickers:
        try:
            histories[ticker] = price_store.read(ticker)
            features_by_ticker[ticker] = feature_store.read(ticker)
        except FileNotFoundError:
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def _fold_metrics(pred, actual, dates):
    gated = simulate_trades(pred, actual, threshold=FIT_THRESHOLD)
    return {
        "mae": float(np.mean(np.abs(pred - actual))),
        "acc": float(np.mean(np.sign(pred) == np.sign(actual))),
        "rank_ic": rank_ic(pred, actual, dates),
        "gated_n": gated["n_trades"],
        "gated_hit": gated["hit_rate"],
        "gated_avg": gated["avg_return"],
    }


def main() -> None:
    started = time.time()
    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout]
    logger.info("train tickers=%s holdout=%s (holdout untouched)", len(train_tickers), len(holdout))
    pooled = _load_pooled(train_tickers, PriceStore(), FeatureStore())
    excluded = pruned_features()
    logger.info("rows=%s (%.1fs)", len(pooled), time.time() - started)

    splits = walk_forward_splits(pooled["date"], n_splits=N_SPLITS)
    baseline_folds = []
    stacked_folds = []
    for fold, (train_mask, test_mask) in enumerate(splits, start=1):
        train_frame = pooled[train_mask]
        test_frame = pooled[test_mask]
        t0 = time.time()
        estimators = fit_stacked_svm_estimators(train_frame, excluded_features=excluded)
        t_svm = time.time()
        logger.info(
            "fold %s fitted %s SVM estimators (train=%s, %.1fs)",
            fold,
            len(estimators),
            len(train_frame),
            t_svm - t0,
        )

        baseline = train_lightgbm(train_frame, excluded_features=excluded)
        stacked_train = attach_stacked_svm_columns(train_frame, estimators)
        stacked_test = attach_stacked_svm_columns(test_frame, estimators)
        # Train-side stacked columns are in-sample (same-fold SVM). LightGBM
        # only *sees* them at fit time so it has a column to split on; the
        # scored test columns are OOF (SVM fit on earlier dates). Do not
        # read train-side stacked metrics -- they leak.
        stacked = train_lightgbm(stacked_train, excluded_features=excluded)

        actual = test_frame[LABEL_COLUMN]
        dates = test_frame["date"]
        baseline_folds.append(_fold_metrics(pd.Series(predict(baseline, test_frame), index=test_frame.index), actual, dates))
        stacked_folds.append(_fold_metrics(pd.Series(predict(stacked, stacked_test), index=test_frame.index), actual, dates))
        logger.info(
            "fold %s lightgbm baseline vs stacked (%.1fs svm, %.1fs total)",
            fold,
            t_svm - t0,
            time.time() - t0,
        )

    def _summarize(name, folds):
        accs = [f["acc"] for f in folds]
        ics = [f["rank_ic"] for f in folds]
        logger.info(
            "%s mean mae=%.5f acc=%.6f rank_ic=%.6f gated_n=%.1f gated_hit=%.6f gated_avg=%.6f",
            name,
            float(np.mean([f["mae"] for f in folds])),
            float(np.mean(accs)),
            float(np.mean(ics)),
            float(np.mean([f["gated_n"] for f in folds])),
            float(np.nanmean([f["gated_hit"] for f in folds])),
            float(np.nanmean([f["gated_avg"] for f in folds])),
        )
        logger.info(
            "  %s acc by fold: %s | rank_ic by fold: %s",
            name,
            [round(a, 4) for a in accs],
            [round(i, 4) for i in ics],
        )
        return accs, ics

    base_acc, base_ic = _summarize("baseline", baseline_folds)
    stack_acc, stack_ic = _summarize("stacked", stacked_folds)
    acc_wins = sum(s > b for s, b in zip(stack_acc, base_acc))
    ic_wins = sum(s > b for s, b in zip(stack_ic, base_ic))
    logger.info(
        "stacked vs baseline: acc wins %s/4, rank_ic wins %s/4 (bar is >=3/4 both)",
        acc_wins,
        ic_wins,
    )
    logger.info("stacked columns=%s", STACKED_SVM_COLUMNS)
    logger.info("done %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
