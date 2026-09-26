"""Solo-vs-blend evaluation for the LinearSVR candidate family.

Same empirical bar random_forest, neural_net, and ridge were held to (see
tune_experiment.py's weight search): every solo is always in the candidate
list, so this can never recommend a blend worse than the best solo already in
production. Ridge rides along as the incumbent linear lens -- if SVR's
epsilon-insensitive loss doesn't separate it from ridge here, there is no
reason to keep a second linear family at all.

Each model is fit ONCE per walk-forward fold; every weight combo is pure
arithmetic over the cached per-fold predictions. Holdout tickers are never
touched -- this decides on validation folds only, and anything that wins goes
through a real run_training + holdout pass before production.

Does not write production pickles or touch PrunedFeatureStore.

Run: bazelisk run //python/stock_picker/training:svr_search
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
from stock_picker.training.model import predict, train_lightgbm, train_ridge, train_svr
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

logger = get_logger(__name__)

N_SPLITS = 4
FIT_THRESHOLD = 0.005

# (lightgbm, svr, ridge). Solos first -- the guarantee that the search can't
# pick a blend worse than the incumbent solo LightGBM.
WEIGHT_CANDIDATES = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (3, 1, 0),
    (1, 1, 0),
    (1, 3, 0),
    (3, 0, 1),
    (3, 1, 1),
    (1, 1, 1),
]


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
    fold_cache = []
    for fold, (train_mask, test_mask) in enumerate(splits, start=1):
        train_frame = pooled[train_mask]
        test_frame = pooled[test_mask]
        t0 = time.time()
        lgbm = train_lightgbm(train_frame, excluded_features=excluded)
        svr = train_svr(train_frame, excluded_features=excluded)
        ridge = train_ridge(train_frame, excluded_features=excluded)
        fold_cache.append(
            {
                "actual": test_frame[LABEL_COLUMN],
                "dates": test_frame["date"],
                "lgbm": pd.Series(predict(lgbm, test_frame), index=test_frame.index),
                "svr": pd.Series(predict(svr, test_frame), index=test_frame.index),
                "ridge": pd.Series(predict(ridge, test_frame), index=test_frame.index),
            }
        )
        logger.info("fold %s fitted lightgbm+svr+ridge (train=%s, %.1fs)", fold, len(train_frame), time.time() - t0)

    rows = []
    for w_lgbm, w_svr, w_ridge in WEIGHT_CANDIDATES:
        total = w_lgbm + w_svr + w_ridge
        maes, accs, ics, gated = [], [], [], []
        for fold in fold_cache:
            blended = (fold["lgbm"] * w_lgbm + fold["svr"] * w_svr + fold["ridge"] * w_ridge) / total
            actual = fold["actual"]
            maes.append(float(np.mean(np.abs(blended - actual))))
            accs.append(float(np.mean(np.sign(blended) == np.sign(actual))))
            ics.append(rank_ic(blended, actual, fold["dates"]))
            gated.append(simulate_trades(blended, actual, threshold=FIT_THRESHOLD))
        rows.append(
            {
                "weights (lgbm,svr,ridge)": f"({w_lgbm},{w_svr},{w_ridge})",
                "mae": float(np.mean(maes)),
                "acc": float(np.mean(accs)),
                "rank_ic": float(np.mean(ics)),
                "gated_n": float(np.mean([g["n_trades"] for g in gated])),
                "gated_hit": float(np.nanmean([g["hit_rate"] for g in gated])),
                "gated_avg": float(np.nanmean([g["avg_return"] for g in gated])),
            }
        )

    summary = pd.DataFrame(rows).sort_values("acc", ascending=False)
    logger.info("mean across %s folds:\n%s", N_SPLITS, summary.to_string(index=False))
    logger.info("done %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
