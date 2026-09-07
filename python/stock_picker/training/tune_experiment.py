"""One-off tuning pass for the day-session-return model: prunes low-value
features (through the real PrunedFeatureStore, same as the Registry UI would),
then searches LightGBM/RandomForest hyperparameters and ensemble weights.

Tuning only ever looks at walk-forward validation folds within the training
tickers -- the held-out tickers are evaluated exactly once, at the very end,
so they stay a genuine unseen-generalization check instead of something the
search implicitly overfits to.

Weight search reuses each fold's already-fitted model predictions rather than
retraining per weight candidate -- blending is just a weighted average, so
there's no reason to refit LightGBM/RandomForest seven more times to try
seven weight ratios.

Not wired into any BUILD-permanent workflow -- a throwaway research script,
run directly via `bazel run //python/stock_picker/training:tune_experiment`.
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pandas as pd

from stock_picker.features.catalog import correlation_matrix, top_correlated_pairs
from stock_picker.features.catalog_loader import feature_tables
from stock_picker.features.pruning import pruned_features
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import sweep_thresholds
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.ensemble import ModelSpec, evaluate_ensemble, predict_ensemble, train_ensemble
from stock_picker.training.importance import model_type_importance
from stock_picker.training.model import (
    NEURAL_NET_DEFAULT_PARAMS,
    RIDGE_DEFAULT_PARAMS,
    predict,
    train_lightgbm,
    train_neural_net,
    train_random_forest,
    train_ridge,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits
from stock_picker.training.train import run_walk_forward

# bazel run's stdout is fully buffered (not a TTY) -- flush every print so
# progress is visible live instead of arriving in one lump at process exit.
print = functools.partial(print, flush=True)

N_SPLITS = 4
NEGLIGIBLE_PCT = 0.5
REDUNDANCY_R = 0.97


def load_pooled(tickers, price_store, feature_store):
    histories, features_by_ticker = {}, {}
    for t in tickers:
        try:
            histories[t] = price_store.read(t)
            features_by_ticker[t] = feature_store.read(t)
        except FileNotFoundError:
            continue
    return build_pooled_dataset(histories, features_by_ticker)


def evaluate_specs(pooled_train, specs, n_splits=N_SPLITS):
    """Fast walk-forward evaluation (no MLflow) for tuning -- average fold
    MAE/directional accuracy. Never touches the held-out tickers."""
    splits = walk_forward_splits(pooled_train["date"], n_splits=n_splits)
    fold_metrics = []
    for train_mask, test_mask in splits:
        ensemble = train_ensemble(pooled_train[train_mask], specs)
        fold_metrics.append(evaluate_ensemble(ensemble, pooled_train[test_mask]))
    avg_mae = sum(m.mae for m in fold_metrics) / len(fold_metrics)
    avg_acc = sum(m.directional_accuracy for m in fold_metrics) / len(fold_metrics)
    return avg_mae, avg_acc, fold_metrics


def prune_low_value_features(pooled_train, excluded):
    """Fits one diagnostic LightGBM+RF pass on all train data to rank
    importance, cross-references Registry's own correlation view for
    redundant pairs, and actually prunes through PrunedFeatureStore --
    same mechanism/records the UI's prune button produces."""
    print("\n=== Fitting diagnostic models on full train set for importance ranking ===")
    lgbm = train_lightgbm(pooled_train, excluded_features=excluded)
    rf = train_random_forest(pooled_train, excluded_features=excluded)
    lgbm_imp = model_type_importance(lgbm)
    rf_imp = model_type_importance(rf)
    blended = {f: (lgbm_imp.get(f, 0) + rf_imp.get(f, 0)) / 2 for f in lgbm_imp}

    corr = correlation_matrix(feature_tables())
    pairs = top_correlated_pairs(corr, n=200)

    store = PrunedFeatureStore()
    already_pruned = set(excluded)
    newly_pruned = []

    # Redundancy first: for each highly-correlated pair, drop whichever side
    # has lower importance (keep the more useful of the two).
    handled = set()
    for pair in pairs:
        if abs(pair["correlation"]) < REDUNDANCY_R:
            break
        a, b = pair["a"], pair["b"]
        if a in already_pruned or b in already_pruned or a in handled or b in handled:
            continue
        loser = a if blended.get(a, 0) <= blended.get(b, 0) else b
        reason = f"high correlation to {b if loser == a else a} (r={pair['correlation']:.3f})"
        store.prune(loser, reason=reason)
        already_pruned.add(loser)
        newly_pruned.append((loser, reason))
        handled.add(a)
        handled.add(b)

    # Then negligible-importance features that survived the redundancy pass.
    for feature, pct in sorted(blended.items(), key=lambda kv: kv[1]):
        if feature in already_pruned:
            continue
        if pct < NEGLIGIBLE_PCT:
            reason = f"negligible importance ({pct:.2f}%)"
            store.prune(feature, reason=reason)
            already_pruned.add(feature)
            newly_pruned.append((feature, reason))

    print(f"Pruned {len(newly_pruned)} additional features:")
    for feature, reason in newly_pruned:
        print(f"  - {feature}: {reason}")
    return pruned_features()


# Deliberately modest grids, not a full cartesian search -- each candidate is
# a full 4-fold walk-forward refit, and LightGBM/RandomForest at this row
# count (~100k+) both scale noticeably with depth/tree count. RandomForest in
# particular never goes past a few hundred shallow-ish trees here: unlimited
# depth on this many rows is the single slowest thing this script could do.
#
# Neither library parallelizes by default in this environment (confirmed by
# a run whose cumulative CPU time tracked wall-clock time 1:1 -- i.e. one
# core, the whole time, on a 14-core machine). Both get explicit thread
# counts below rather than trusting an auto-detect that clearly isn't firing.
N_CORES = os.cpu_count() or 4

# Round 1 (depth/leaves/min-data only) picked num_leaves=15/max_depth=4/
# min_data_in_leaf=30 as the winner (see git history for the full round-1
# grid) -- round 2 holds that shape fixed and varies the regularization
# knobs round 1 never touched (feature/bagging fraction, L2), plus a couple
# of RF configs with meaningfully more trees now that n_jobs actually works.
LGBM_CANDIDATES = [
    {"num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 30, "learning_rate": 0.05, "num_threads": N_CORES},
    {
        "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 30, "learning_rate": 0.05,
        "feature_fraction": 0.8, "num_threads": N_CORES,
    },
    {
        "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 30, "learning_rate": 0.05,
        "bagging_fraction": 0.8, "bagging_freq": 5, "num_threads": N_CORES,
    },
    {
        "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 30, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l2": 1.0,
        "num_threads": N_CORES,
    },
    {
        "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 30, "learning_rate": 0.03,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "num_threads": N_CORES,
    },
]
RF_CANDIDATES = [
    {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 20, "n_jobs": -1},
    {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 20, "n_jobs": -1},
    {"n_estimators": 500, "max_depth": 6, "min_samples_leaf": 30, "max_features": 0.5, "n_jobs": -1},
]


def tune_hyperparams(pooled_train, excluded):
    print("\n=== Tuning LightGBM ===")
    best_lgbm, best_lgbm_acc = None, -1
    for params in LGBM_CANDIDATES:
        t0 = time.time()
        mae, acc, _ = evaluate_specs(pooled_train, [ModelSpec("lightgbm", params=params, excluded_features=excluded)])
        print(f"  {params} -> MAE={mae:.5f} acc={acc:.4f} ({time.time() - t0:.1f}s)")
        if acc > best_lgbm_acc:
            best_lgbm, best_lgbm_acc = params, acc

    print("\n=== Tuning Random Forest ===")
    best_rf, best_rf_acc = None, -1
    for params in RF_CANDIDATES:
        t0 = time.time()
        mae, acc, _ = evaluate_specs(pooled_train, [ModelSpec("random_forest", params=params, excluded_features=excluded)])
        print(f"  {params} -> MAE={mae:.5f} acc={acc:.4f} ({time.time() - t0:.1f}s)")
        if acc > best_rf_acc:
            best_rf, best_rf_acc = params, acc

    print(f"\nBest LightGBM: {best_lgbm} (acc={best_lgbm_acc:.4f})")
    print(f"Best RandomForest: {best_rf} (acc={best_rf_acc:.4f})")
    return best_lgbm, best_rf


# 3-way (lightgbm, random_forest, neural_net). Includes every solo/pairwise
# 4-way (lightgbm, random_forest, neural_net, ridge). Includes every solo --
# most importantly (1,0,0,0) -- so the search can never recommend a blend
# worse than the best solo model already in production; that's how
# random_forest's and neural_net's exclusion from DEFAULT_MODEL_SPECS was
# decided empirically rather than assumed, and this decides ridge's fate the
# same way. Training all 4 models per fold is the fixed cost regardless of
# how many weight combos get tried afterward (each combo is just arithmetic
# over cached per-fold predictions), so this list can afford to be broad.
WEIGHT_CANDIDATES = [
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
]


def tune_weights(pooled_train, excluded, lgbm_params, rf_params, nn_params, ridge_params, n_splits=N_SPLITS):
    """Fits each model ONCE per fold, caches all four models' raw predictions
    and the fold's actual labels, then scores every weight candidate as pure
    arithmetic over those cached arrays -- no retraining per weight."""
    print("\n=== Tuning ensemble weights (reusing cached per-fold predictions) ===")
    splits = walk_forward_splits(pooled_train["date"], n_splits=n_splits)
    fold_cache = []
    for train_mask, test_mask in splits:
        train_frame = pooled_train[train_mask]
        test_frame = pooled_train[test_mask]
        lgbm = train_lightgbm(train_frame, params=lgbm_params, excluded_features=excluded)
        rf = train_random_forest(train_frame, params=rf_params, excluded_features=excluded)
        nn = train_neural_net(train_frame, params=nn_params, excluded_features=excluded)
        ridge = train_ridge(train_frame, params=ridge_params, excluded_features=excluded)
        fold_cache.append(
            {
                "actual": test_frame[LABEL_COLUMN].to_numpy(),
                "lgbm_pred": predict(lgbm, test_frame),
                "rf_pred": predict(rf, test_frame),
                "nn_pred": predict(nn, test_frame),
                "ridge_pred": predict(ridge, test_frame),
            }
        )

    best_weights, best_acc = None, -1
    for w_lgbm, w_rf, w_nn, w_ridge in WEIGHT_CANDIDATES:
        total = w_lgbm + w_rf + w_nn + w_ridge
        maes, accs = [], []
        for fold in fold_cache:
            if total == 0:
                continue
            blended = (
                fold["lgbm_pred"] * w_lgbm
                + fold["rf_pred"] * w_rf
                + fold["nn_pred"] * w_nn
                + fold["ridge_pred"] * w_ridge
            ) / total
            actual = fold["actual"]
            maes.append(float(np.mean(np.abs(blended - actual))))
            accs.append(float(np.mean(np.sign(blended) == np.sign(actual))))
        mae, acc = sum(maes) / len(maes), sum(accs) / len(accs)
        print(
            f"  weights lightgbm={w_lgbm} random_forest={w_rf} neural_net={w_nn} ridge={w_ridge} "
            f"-> MAE={mae:.5f} acc={acc:.4f}"
        )
        if acc > best_acc:
            best_weights, best_acc = (w_lgbm, w_rf, w_nn, w_ridge), acc
    print(
        f"\nBest weights: lightgbm={best_weights[0]} random_forest={best_weights[1]} "
        f"neural_net={best_weights[2]} ridge={best_weights[3]} (acc={best_acc:.4f})"
    )
    return best_weights


def main() -> None:
    t0 = time.time()
    tickers = UniverseStore().active_tickers()
    holdout_ticker_set = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout_ticker_set]
    holdout_tickers = [t for t in tickers if t in holdout_ticker_set]
    print(f"universe: {len(tickers)} tickers -- {len(train_tickers)} train / {len(holdout_tickers)} holdout")

    price_store = PriceStore()
    feature_store = FeatureStore()

    print("loading pooled train dataset...")
    pooled_train = load_pooled(train_tickers, price_store, feature_store)
    print(f"pooled train rows: {len(pooled_train)} ({time.time() - t0:.1f}s elapsed)")

    excluded = pruned_features()
    print(f"currently pruned ({len(excluded)}): {sorted(excluded)}")

    baseline_specs = [
        ModelSpec("lightgbm", excluded_features=excluded),
        ModelSpec("random_forest", excluded_features=excluded),
        ModelSpec("neural_net", excluded_features=excluded),
        ModelSpec("ridge", excluded_features=excluded),
    ]
    mae, acc, _ = evaluate_specs(pooled_train, baseline_specs)
    print("\n=== BASELINE (current pruned set, equal weight, default params) ===")
    print(f"avg fold MAE={mae:.5f} avg directional accuracy={acc:.4f}")

    excluded = prune_low_value_features(pooled_train, excluded)
    mae, acc, _ = evaluate_specs(
        pooled_train,
        [
            ModelSpec("lightgbm", excluded_features=excluded),
            ModelSpec("random_forest", excluded_features=excluded),
            ModelSpec("neural_net", excluded_features=excluded),
            ModelSpec("ridge", excluded_features=excluded),
        ],
    )
    print(f"\n=== AFTER PRUNING ({len(excluded)} excluded) ===")
    print(f"avg fold MAE={mae:.5f} avg directional accuracy={acc:.4f}")

    lgbm_params, rf_params = tune_hyperparams(pooled_train, excluded)
    # Neither re-tuned here -- both NEURAL_NET_DEFAULT_PARAMS and
    # RIDGE_DEFAULT_PARAMS were already chosen conservatively against this
    # same feature-to-sample ratio when each was first added (see model.py);
    # this run's job is only to decide *whether* ridge earns ensemble weight,
    # the same question already settled for random_forest and neural_net.
    nn_params = NEURAL_NET_DEFAULT_PARAMS
    ridge_params = RIDGE_DEFAULT_PARAMS
    weights = tune_weights(pooled_train, excluded, lgbm_params, rf_params, nn_params, ridge_params)

    final_specs = [
        ModelSpec("lightgbm", params=lgbm_params, excluded_features=excluded, weight=weights[0]),
        ModelSpec("random_forest", params=rf_params, excluded_features=excluded, weight=weights[1]),
        ModelSpec("neural_net", params=nn_params, excluded_features=excluded, weight=weights[2]),
        ModelSpec("ridge", params=ridge_params, excluded_features=excluded, weight=weights[3]),
    ]
    mae, acc, _ = evaluate_specs(pooled_train, final_specs)
    print("\n=== FINAL TUNED CONFIG (validation folds only) ===")
    print(f"avg fold MAE={mae:.5f} avg directional accuracy={acc:.4f}")

    print("\n=== Final walk-forward run (with MLflow logging) + genuine holdout check ===")
    fold_results = run_walk_forward(pooled_train, specs=final_specs)
    for r in fold_results:
        print(f"  fold {r.fold}: {r.metrics}")
    final_ensemble = fold_results[-1].model

    pooled_holdout = load_pooled(holdout_tickers, price_store, feature_store)
    holdout_metrics = evaluate_ensemble(final_ensemble, pooled_holdout)
    print(f"\nHOLDOUT (never-seen tickers, {len(holdout_tickers)} tickers): {holdout_metrics}")

    predicted = pd.Series(predict_ensemble(final_ensemble, pooled_holdout), index=pooled_holdout.index)
    actual = pooled_holdout[LABEL_COLUMN]
    sweep = sweep_thresholds(predicted, actual)
    print("\nThreshold sweep on holdout:")
    print(sweep.to_string(index=False))

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    print(f"\nFinal config -- lightgbm params={lgbm_params} weight={weights[0]}")
    print(f"Final config -- random_forest params={rf_params} weight={weights[1]}")
    print(f"Final config -- neural_net params={nn_params} weight={weights[2]}")
    print(f"Final config -- ridge params={ridge_params} weight={weights[3]}")
    print(f"Final pruned set ({len(excluded)}): {sorted(excluded)}")


if __name__ == "__main__":
    main()
