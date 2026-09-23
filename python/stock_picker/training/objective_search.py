"""Walk-forward bake-off of LightGBM objectives on our open→close label.

Does not write production pickles. Production Fit stays MAE-regression;
Rank stays lambdarank until a candidate beats both on Rank IC *and* top-20
session return.

Candidates:
  mae         -- current Fit
  huber       -- less wrecked by +10% outliers than MAE/MSE
  lambdarank  -- current Rank
  rank_xendcg -- listwise NDCG, often more stable than lambdarank

Run: bazelisk run //python/stock_picker/training:objective_search
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from stock_picker.features.pruning import pruned_features
from stock_picker.log import get_logger
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import rank_ic, simulate_top_k, simulate_trades
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.model import (
    DEFAULT_NUM_BOOST_ROUND,
    LIGHTGBM_DEFAULT_PARAMS,
    LIGHTGBM_RANK_GRADES,
    feature_columns,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

logger = get_logger(__name__)

N_SPLITS = 4
TOP_K = 20
FIT_THRESHOLD = 0.005
CANDIDATES = ("mae", "huber", "lambdarank", "rank_xendcg")


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


def _relevance(frame: pd.DataFrame, grades: int = LIGHTGBM_RANK_GRADES) -> pd.Series:
    def _grades(day: pd.Series) -> pd.Series:
        out = pd.Series(0, index=day.index, dtype=int)
        ups = day[day > 0]
        if ups.empty:
            return out
        n = min(grades, max(2, int(ups.nunique())))
        buckets = pd.qcut(ups.rank(method="first"), n, labels=False, duplicates="drop")
        out.loc[ups.index] = buckets.astype(int) + 1
        return out

    return frame.groupby("date")[LABEL_COLUMN].transform(_grades).fillna(0).astype(int)


def _fit(objective: str, train_frame: pd.DataFrame, columns: list[str]) -> lgb.Booster:
    params = {
        **LIGHTGBM_DEFAULT_PARAMS,
        "objective": objective,
        "verbosity": -1,
    }
    if objective in {"lambdarank", "rank_xendcg"}:
        sorted_frame = train_frame.sort_values("date")
        relevance = _relevance(sorted_frame)
        group_sizes = sorted_frame.groupby("date").size().to_numpy()
        dataset = lgb.Dataset(sorted_frame[columns], label=relevance, group=group_sizes)
        params["metric"] = "ndcg"
        params["eval_at"] = [5, 20]
    else:
        dataset = lgb.Dataset(train_frame[columns], label=train_frame[LABEL_COLUMN])
        params["metric"] = "mae" if objective == "mae" else "huber"
        if objective == "huber":
            params["alpha"] = 0.9
    return lgb.train(params, dataset, num_boost_round=DEFAULT_NUM_BOOST_ROUND)


def _score_fold(pred: pd.Series, test_frame: pd.DataFrame) -> dict:
    actual = test_frame[LABEL_COLUMN]
    dates = test_frame["date"]
    top = simulate_top_k(pred, actual, dates, k=TOP_K)
    gated = simulate_trades(pred, actual, threshold=FIT_THRESHOLD, n_days=int(dates.nunique()))
    return {
        "rank_ic": rank_ic(pred, actual, dates),
        "top20_hit": top["hit_rate"],
        "top20_avg": top["avg_return"],
        "fit_n": gated["n_trades"],
        "fit_hit": gated["hit_rate"],
        "fit_avg": gated["avg_return"],
    }


def main() -> None:
    started = time.time()
    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout]
    logger.info("train tickers=%s holdout=%s", len(train_tickers), len(holdout))
    pooled = _load_pooled(train_tickers, PriceStore(), FeatureStore())
    excluded = pruned_features()
    columns = feature_columns(pooled, excluded)
    logger.info("rows=%s features=%s (%.1fs)", len(pooled), len(columns), time.time() - started)

    splits = walk_forward_splits(pooled["date"], n_splits=N_SPLITS)
    by_objective: dict[str, list[dict]] = {name: [] for name in CANDIDATES}

    for fold, (train_mask, test_mask) in enumerate(splits, start=1):
        train_frame = pooled[train_mask]
        test_frame = pooled[test_mask]
        logger.info("fold %s train=%s test=%s", fold, len(train_frame), len(test_frame))
        for name in CANDIDATES:
            t0 = time.time()
            booster = _fit(name, train_frame, columns)
            pred = pd.Series(booster.predict(test_frame[columns]), index=test_frame.index)
            metrics = _score_fold(pred, test_frame)
            metrics["seconds"] = time.time() - t0
            by_objective[name].append(metrics)
            logger.info(
                "  %s  IC=%.4f top20_hit=%.3f top20_avg=%.4f fit_n=%s fit_hit=%s (%.1fs)",
                name,
                metrics["rank_ic"],
                metrics["top20_hit"],
                metrics["top20_avg"],
                metrics["fit_n"],
                metrics["fit_hit"],
                metrics["seconds"],
            )

    rows = []
    for name, folds in by_objective.items():
        frame = pd.DataFrame(folds)
        rows.append(
            {
                "objective": name,
                "rank_ic": float(frame["rank_ic"].mean()),
                "top20_hit": float(frame["top20_hit"].mean()),
                "top20_avg": float(frame["top20_avg"].mean()),
                "fit_n": float(frame["fit_n"].mean()),
                "fit_hit": float(np.nanmean(frame["fit_hit"])),
                "fit_avg": float(np.nanmean(frame["fit_avg"])),
            }
        )
    summary = pd.DataFrame(rows).sort_values("top20_avg", ascending=False)
    logger.info("mean across %s folds:\n%s", N_SPLITS, summary.to_string(index=False))
    logger.info("done %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
