"""Bounded ~1-hour search for a better day-session ensemble after the
open-known recency features landed.

Does NOT rewrite PrunedFeatureStore -- that store is the Registry UI's
record, and tune_experiment.py's prune pass would mutate it. This script
only searches model families and blend weights on the current pruned set
plus whatever new columns survived pruning (the 24 recency columns).

Walk-forward folds decide the winner. Holdout tickers are scored once, at
the end, on the winner only. The winner is then persisted through
run_training() so ModelStore / run history / live scoring all see it.
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pandas as pd

from stock_picker.features.pruning import pruned_features
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import ModelSpec, evaluate_ensemble, predict_ensemble
from stock_picker.training.main import _load_pooled_dataset, run_training
from stock_picker.training.model import (
    LIGHTGBM_DEFAULT_PARAMS,
    RANDOM_FOREST_DEFAULT_PARAMS,
    RIDGE_DEFAULT_PARAMS,
    predict,
    train_lightgbm,
    train_random_forest,
    train_ridge,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

print = functools.partial(print, flush=True)

N_SPLITS = 4
N_CORES = os.cpu_count() or 4

# Current production shape, plus a couple of nearby regularizations. Modest
# on purpose: each candidate is a full 4-fold refit.
LGBM_CANDIDATES = [
    {**LIGHTGBM_DEFAULT_PARAMS, "num_threads": N_CORES},
    {
        **LIGHTGBM_DEFAULT_PARAMS,
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_threads": N_CORES,
    },
    {
        **LIGHTGBM_DEFAULT_PARAMS,
        "lambda_l2": 1.0,
        "num_threads": N_CORES,
    },
    {
        **LIGHTGBM_DEFAULT_PARAMS,
        "num_leaves": 63,
        "max_depth": 6,
        "learning_rate": 0.02,
        "num_threads": N_CORES,
    },
]
RF_PARAMS = {**RANDOM_FOREST_DEFAULT_PARAMS, "n_jobs": -1}
RIDGE_PARAMS = dict(RIDGE_DEFAULT_PARAMS)

# Includes every solo so a blend cannot win unless it beats the best solo.
WEIGHT_CANDIDATES = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (2, 1, 0),
    (2, 0, 1),
    (1, 1, 1),
    (2, 1, 1),
]


def _score(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - actual)))
    acc = float(np.mean(np.sign(pred) == np.sign(actual)))
    return mae, acc


def evaluate_specs(pooled_train: pd.DataFrame, specs: list[ModelSpec]) -> tuple[float, float]:
    splits = walk_forward_splits(pooled_train["date"], n_splits=N_SPLITS)
    maes, accs = [], []
    for train_mask, test_mask in splits:
        from stock_picker.training.ensemble import train_ensemble

        ensemble = train_ensemble(pooled_train[train_mask], specs)
        metrics = evaluate_ensemble(ensemble, pooled_train[test_mask])
        maes.append(metrics.mae)
        accs.append(metrics.directional_accuracy)
    return sum(maes) / len(maes), sum(accs) / len(accs)


def main() -> None:
    t0 = time.time()
    excluded = pruned_features()
    tickers = UniverseStore().active_tickers()
    holdout_set = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout_set]
    holdout_tickers = [t for t in tickers if t in holdout_set]
    price_store = PriceStore()
    feature_store = FeatureStore()

    print(f"loading train pool ({len(train_tickers)} tickers, {len(excluded)} pruned)...")
    pooled_train = _load_pooled_dataset(train_tickers, price_store, feature_store)
    n_open = sum(1 for c in pooled_train.columns if c.endswith("_seasonality") and "open" in c)
    print(
        f"train rows={len(pooled_train)} cols={pooled_train.shape[1]} "
        f"open-known seasonality cols present={n_open} ({time.time() - t0:.1f}s)"
    )

    print("\n=== LightGBM hyperparams (walk-forward, train tickers only) ===")
    best_lgbm, best_lgbm_acc, best_lgbm_mae = None, -1.0, 1.0
    for params in LGBM_CANDIDATES:
        started = time.time()
        mae, acc = evaluate_specs(
            pooled_train, [ModelSpec("lightgbm", params=params, excluded_features=excluded)]
        )
        print(f"  leaves={params['num_leaves']} depth={params['max_depth']} lr={params['learning_rate']} "
              f"l2={params.get('lambda_l2', 0)} -> MAE={mae:.5f} acc={acc:.4f} ({time.time() - started:.1f}s)")
        if acc > best_lgbm_acc or (acc == best_lgbm_acc and mae < best_lgbm_mae):
            best_lgbm, best_lgbm_acc, best_lgbm_mae = params, acc, mae
    print(f"Best LightGBM acc={best_lgbm_acc:.4f} MAE={best_lgbm_mae:.5f}")

    print("\n=== Cache per-fold LGBM / RF / Ridge predictions ===")
    splits = walk_forward_splits(pooled_train["date"], n_splits=N_SPLITS)
    fold_cache = []
    for i, (train_mask, test_mask) in enumerate(splits, start=1):
        started = time.time()
        train_frame = pooled_train[train_mask]
        test_frame = pooled_train[test_mask]
        lgbm = train_lightgbm(train_frame, params=best_lgbm, excluded_features=excluded)
        rf = train_random_forest(train_frame, params=RF_PARAMS, excluded_features=excluded)
        ridge = train_ridge(train_frame, params=RIDGE_PARAMS, excluded_features=excluded)
        fold_cache.append(
            {
                "actual": test_frame[LABEL_COLUMN].to_numpy(),
                "lgbm": predict(lgbm, test_frame),
                "rf": predict(rf, test_frame),
                "ridge": predict(ridge, test_frame),
            }
        )
        print(f"  fold {i} cached ({time.time() - started:.1f}s)")

    print("\n=== Ensemble weights ===")
    best_weights, best_acc, best_mae = None, -1.0, 1.0
    for w_lgbm, w_rf, w_ridge in WEIGHT_CANDIDATES:
        total = w_lgbm + w_rf + w_ridge
        maes, accs = [], []
        for fold in fold_cache:
            blended = (fold["lgbm"] * w_lgbm + fold["rf"] * w_rf + fold["ridge"] * w_ridge) / total
            mae, acc = _score(fold["actual"], blended)
            maes.append(mae)
            accs.append(acc)
        mae, acc = sum(maes) / len(maes), sum(accs) / len(accs)
        print(f"  lgbm={w_lgbm} rf={w_rf} ridge={w_ridge} -> MAE={mae:.5f} acc={acc:.4f}")
        if acc > best_acc or (acc == best_acc and mae < best_mae):
            best_weights, best_acc, best_mae = (w_lgbm, w_rf, w_ridge), acc, mae
    print(f"Best weights {best_weights} acc={best_acc:.4f} MAE={best_mae:.5f}")

    winner_specs = []
    names = ("lightgbm", "random_forest", "ridge")
    params_by_name = {"lightgbm": best_lgbm, "random_forest": RF_PARAMS, "ridge": RIDGE_PARAMS}
    for name, weight in zip(names, best_weights):
        if weight <= 0:
            continue
        winner_specs.append(
            ModelSpec(name, params=params_by_name[name], excluded_features=excluded, weight=float(weight))
        )
    print("\nWinner specs:")
    for spec in winner_specs:
        print(f"  {spec.model_type} weight={spec.weight} params={spec.params}")

    print("\n=== Persist via run_training (holdout scored there too) ===")
    summary = run_training(model_specs=winner_specs)
    print(f"holdout: {summary.holdout_metrics}")
    if summary.threshold_sweep:
        sweep = pd.DataFrame(summary.threshold_sweep)
        print("threshold sweep:")
        print(sweep.to_string(index=False))
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
