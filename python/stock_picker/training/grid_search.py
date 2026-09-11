"""Denser LightGBM grid + two ensemble experiments on the current feature store.

Does NOT rewrite PrunedFeatureStore. Holdout tickers are scored only after
walk-forward picks a candidate. Production `latest` is overwritten only if
holdout directional accuracy beats the current solo LightGBM baseline.

Ensemble experiments (walk-forward, never shuffled):
- return-average: fixed weights, including every solo
- rank-average: mean of within-day percentile ranks (Rank IC / top-K, not MAE)
- learned linear blend: w fitted on strictly earlier folds' OOF predictions
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
from stock_picker.training.backtest import rank_ic, sweep_thresholds
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import ModelSpec, evaluate_ensemble, predict_ensemble
from stock_picker.training.main import _load_pooled_dataset, run_training
from stock_picker.training.model import (
    DEFAULT_NUM_BOOST_ROUND,
    LIGHTGBM_DEFAULT_PARAMS,
    RIDGE_DEFAULT_PARAMS,
    predict,
    train_lightgbm,
    train_ridge,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

print = functools.partial(print, flush=True)

N_SPLITS = 4
N_CORES = os.cpu_count() or 4
# Last persisted solo LightGBM on the 24 recency columns (not this Phase-1 set).
CURRENT_HOLDOUT_ACC = 0.5765719978043872
TOP_K = 5

PHASE1_COLUMNS = [
    "gap_seq3_open3_seasonality",
    "gap_trap_open_seasonality",
    "streak_crash_open_seasonality",
    "multi_crash_bounce_open_seasonality",
]


def _lgbm(overrides: dict) -> dict:
    params = {**LIGHTGBM_DEFAULT_PARAMS, "num_threads": N_CORES, **overrides}
    return params


# Each candidate is a full 4-fold walk-forward LightGBM refit. Keep this a
# grid, not a cartesian product -- ~12 shapes, including the production default.
LGBM_CANDIDATES = [
    _lgbm({}),
    _lgbm({"num_boost_round": 200}),
    _lgbm({"num_boost_round": 300, "learning_rate": 0.02}),
    _lgbm({"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05}),
    _lgbm({"num_leaves": 63, "max_depth": 6, "learning_rate": 0.02}),
    _lgbm({"num_leaves": 63, "max_depth": 6, "learning_rate": 0.02, "num_boost_round": 200}),
    _lgbm({"lambda_l2": 1.0}),
    _lgbm({"lambda_l2": 5.0}),
    _lgbm({"feature_fraction": 0.6}),
    _lgbm({"feature_fraction": 1.0}),
    _lgbm({"min_data_in_leaf": 50}),
    _lgbm({"num_leaves": 31, "max_depth": 7, "learning_rate": 0.03}),
]


def _split_lgbm_params(params: dict) -> tuple[dict, int]:
    clean = dict(params)
    rounds = int(clean.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND))
    return clean, rounds


def _score(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - actual)))
    acc = float(np.mean(np.sign(pred) == np.sign(actual)))
    return mae, acc


def _fit_lgbm(train_frame: pd.DataFrame, params: dict, excluded: set[str]):
    lgbm_params, rounds = _split_lgbm_params(params)
    return train_lightgbm(
        train_frame, params=lgbm_params, num_boost_round=rounds, excluded_features=excluded
    )


def evaluate_lgbm(pooled_train: pd.DataFrame, params: dict, excluded: set[str]) -> tuple[float, float]:
    splits = walk_forward_splits(pooled_train["date"], n_splits=N_SPLITS)
    maes, accs = [], []
    for train_mask, test_mask in splits:
        model = _fit_lgbm(pooled_train[train_mask], params, excluded)
        pred = predict(model, pooled_train[test_mask])
        mae, acc = _score(pooled_train[test_mask][LABEL_COLUMN].to_numpy(), pred)
        maes.append(mae)
        accs.append(acc)
    return sum(maes) / len(maes), sum(accs) / len(accs)


def _within_day_rank(pred: np.ndarray, dates: pd.Series) -> np.ndarray:
    ranks = pd.Series(pred, index=dates.index).groupby(dates).rank(pct=True)
    return ranks.to_numpy()


def _top_k_hit_rate(pred: np.ndarray, actual: np.ndarray, dates: pd.Series, k: int = TOP_K) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual, "date": dates})
    hits = []
    for _, day in frame.groupby("date"):
        if len(day) < k:
            continue
        top = day.nlargest(k, "pred")
        hits.append(float((top["actual"] > 0).mean()))
    return float(np.mean(hits)) if hits else float("nan")


def _describe(params: dict) -> str:
    lgbm_params, rounds = _split_lgbm_params(params)
    return (
        f"leaves={lgbm_params['num_leaves']} depth={lgbm_params['max_depth']} "
        f"lr={lgbm_params['learning_rate']} l2={lgbm_params.get('lambda_l2', 0)} "
        f"ff={lgbm_params.get('feature_fraction')} min_leaf={lgbm_params['min_data_in_leaf']} "
        f"rounds={rounds}"
    )


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
    present_phase1 = [c for c in PHASE1_COLUMNS if c in pooled_train.columns]
    n_open = sum(
        1
        for c in pooled_train.columns
        if c.endswith("_seasonality") and ("open" in c or c.startswith("gap_"))
    )
    print(
        f"train rows={len(pooled_train)} cols={pooled_train.shape[1]} "
        f"open-known seasonality={n_open} phase1={present_phase1} ({time.time() - t0:.1f}s)"
    )
    if len(present_phase1) != len(PHASE1_COLUMNS):
        missing = [c for c in PHASE1_COLUMNS if c not in pooled_train.columns]
        raise SystemExit(f"feature store missing Phase-1 columns {missing} -- rebuild features first")

    print("\n=== LightGBM grid (walk-forward, train tickers only) ===")
    best_lgbm, best_lgbm_acc, best_lgbm_mae = None, -1.0, 1.0
    for params in LGBM_CANDIDATES:
        started = time.time()
        mae, acc = evaluate_lgbm(pooled_train, params, excluded)
        print(f"  {_describe(params)} -> MAE={mae:.5f} acc={acc:.4f} ({time.time() - started:.1f}s)")
        if acc > best_lgbm_acc or (acc == best_lgbm_acc and mae < best_lgbm_mae):
            best_lgbm, best_lgbm_acc, best_lgbm_mae = params, acc, mae
    print(f"Best LightGBM {_describe(best_lgbm)} acc={best_lgbm_acc:.4f} MAE={best_lgbm_mae:.5f}")

    print("\n=== Cache per-fold LightGBM / Ridge predictions ===")
    splits = walk_forward_splits(pooled_train["date"], n_splits=N_SPLITS)
    fold_cache = []
    for i, (train_mask, test_mask) in enumerate(splits, start=1):
        started = time.time()
        train_frame = pooled_train[train_mask]
        test_frame = pooled_train[test_mask]
        lgbm = _fit_lgbm(train_frame, best_lgbm, excluded)
        ridge = train_ridge(train_frame, params=RIDGE_DEFAULT_PARAMS, excluded_features=excluded)
        lgbm_pred = predict(lgbm, test_frame)
        ridge_pred = predict(ridge, test_frame)
        actual = test_frame[LABEL_COLUMN].to_numpy()
        dates = test_frame["date"]
        fold_cache.append(
            {
                "actual": actual,
                "dates": dates,
                "lgbm": lgbm_pred,
                "ridge": ridge_pred,
            }
        )
        print(f"  fold {i} cached ({time.time() - started:.1f}s)")

    print("\n=== Return-average weights ===")
    best_weights, best_acc, best_mae = None, -1.0, 1.0
    for w_lgbm, w_ridge in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)):
        total = w_lgbm + w_ridge
        maes, accs = [], []
        for fold in fold_cache:
            blended = (fold["lgbm"] * w_lgbm + fold["ridge"] * w_ridge) / total
            mae, acc = _score(fold["actual"], blended)
            maes.append(mae)
            accs.append(acc)
        mae, acc = sum(maes) / len(maes), sum(accs) / len(accs)
        print(f"  lgbm={w_lgbm} ridge={w_ridge} -> MAE={mae:.5f} acc={acc:.4f}")
        if acc > best_acc or (acc == best_acc and mae < best_mae):
            best_weights, best_acc, best_mae = (w_lgbm, w_ridge), acc, mae
    print(f"Best return-average {best_weights} acc={best_acc:.4f} MAE={best_mae:.5f}")

    print("\n=== Rank-average (within-day percentile ranks) ===")
    for name, builder in (
        ("lgbm", lambda f: f["lgbm"]),
        ("ridge", lambda f: f["ridge"]),
        ("mean rank", lambda f: (_within_day_rank(f["lgbm"], f["dates"]) + _within_day_rank(f["ridge"], f["dates"])) / 2),
    ):
        ics, topks = [], []
        for fold in fold_cache:
            pred = builder(fold)
            ics.append(rank_ic(pd.Series(pred), pd.Series(fold["actual"]), fold["dates"]))
            topks.append(_top_k_hit_rate(pred, fold["actual"], fold["dates"]))
        print(f"  {name}: Rank IC={np.nanmean(ics):.4f} top-{TOP_K} hit={np.nanmean(topks):.4f}")

    print("\n=== Walk-forward learned blend w*lgbm + (1-w)*ridge ===")
    # Fold 0 has no earlier OOF rows -- skip it for the learned-w average.
    ws = np.round(np.linspace(0.0, 1.0, 11), 2)
    learned_accs, learned_maes, chosen = [], [], []
    for k in range(1, len(fold_cache)):
        prior = fold_cache[:k]
        prior_lgbm = np.concatenate([f["lgbm"] for f in prior])
        prior_ridge = np.concatenate([f["ridge"] for f in prior])
        prior_actual = np.concatenate([f["actual"] for f in prior])
        best_w, best_w_acc, best_w_mae = 1.0, -1.0, 1.0
        for w in ws:
            blended = w * prior_lgbm + (1 - w) * prior_ridge
            mae, acc = _score(prior_actual, blended)
            if acc > best_w_acc or (acc == best_w_acc and mae < best_w_mae):
                best_w, best_w_acc, best_w_mae = float(w), acc, mae
        test = fold_cache[k]
        test_blend = best_w * test["lgbm"] + (1 - best_w) * test["ridge"]
        mae, acc = _score(test["actual"], test_blend)
        learned_accs.append(acc)
        learned_maes.append(mae)
        chosen.append(best_w)
        print(f"  fold {k+1}: w={best_w:.2f} (fit on earlier folds) -> MAE={mae:.5f} acc={acc:.4f}")
    print(
        f"Learned-w mean acc={sum(learned_accs)/len(learned_accs):.4f} "
        f"MAE={sum(learned_maes)/len(learned_maes):.5f} chosen w={chosen}"
    )

    lgbm_params, rounds = _split_lgbm_params(best_lgbm)
    persistable_lgbm_params = {**lgbm_params, "num_boost_round": rounds}
    if best_weights != (1, 0) and best_acc > best_lgbm_acc:
        print("\nWalk-forward prefers a return-average blend over solo LightGBM.")
    else:
        print("\nWalk-forward prefers solo LightGBM.")

    print("\n=== Holdout (never-seen tickers) ===")
    pooled_holdout = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
    train_all = pooled_train
    lgbm_final = _fit_lgbm(train_all, best_lgbm, excluded)
    ridge_final = train_ridge(train_all, params=RIDGE_DEFAULT_PARAMS, excluded_features=excluded)
    lgbm_hat = predict(lgbm_final, pooled_holdout)
    ridge_hat = predict(ridge_final, pooled_holdout)
    actual = pooled_holdout[LABEL_COLUMN].to_numpy()
    dates = pooled_holdout["date"]

    def _report(name: str, pred: np.ndarray) -> float:
        mae, acc = _score(actual, pred)
        ic = rank_ic(pd.Series(pred), pooled_holdout[LABEL_COLUMN], dates)
        topk = _top_k_hit_rate(pred, actual, dates)
        sweep = sweep_thresholds(pd.Series(pred), pooled_holdout[LABEL_COLUMN], n_days=dates.nunique())
        row = sweep.loc[sweep["threshold"] == 0.005].iloc[0]
        print(
            f"  {name}: acc={acc:.4f} MAE={mae:.5f} Rank IC={ic:.4f} "
            f"top-{TOP_K}={topk:.4f} | 0.5% hit={row['hit_rate']:.4f} "
            f"n={int(row['n_trades'])} tot={row['total_return']:.2f}"
        )
        return acc

    solo_acc = _report("solo LightGBM", lgbm_hat)
    _report("solo Ridge", ridge_hat)
    blend_pred = (lgbm_hat + ridge_hat) / 2
    blend_acc = _report("return-avg 50/50", blend_pred)
    rank_blend = (_within_day_rank(lgbm_hat, dates) + _within_day_rank(ridge_hat, dates)) / 2
    # Rank blend has no return units -- Rank IC / top-K only.
    ic = rank_ic(pd.Series(rank_blend), pooled_holdout[LABEL_COLUMN], dates)
    topk = _top_k_hit_rate(rank_blend, actual, dates)
    print(f"  rank-avg LightGBM+Ridge: Rank IC={ic:.4f} top-{TOP_K}={topk:.4f} (no return-threshold)")

    mean_w = float(np.mean(chosen)) if chosen else 1.0
    learned_pred = mean_w * lgbm_hat + (1 - mean_w) * ridge_hat
    learned_acc = _report(f"learned-w w={mean_w:.2f} (mean of fold choices)", learned_pred)

    holdout_scores = {
        "solo LightGBM": solo_acc,
        "return-avg 50/50": blend_acc,
        f"learned-w {mean_w:.2f}": learned_acc,
    }
    winner_name = max(holdout_scores, key=holdout_scores.get)
    winner_acc = holdout_scores[winner_name]
    print(f"\nHoldout winner among these: {winner_name} acc={winner_acc:.4f}")
    print(f"Current production baseline acc={CURRENT_HOLDOUT_ACC:.4f}")

    if winner_acc <= CURRENT_HOLDOUT_ACC:
        print("\nNot persisting -- nothing beat the current production holdout accuracy.")
    elif winner_name == "solo LightGBM":
        print("\nPersisting solo LightGBM via run_training -- holdout beat the previous latest.")
        run_training(
            model_specs=[
                ModelSpec("lightgbm", params=persistable_lgbm_params, excluded_features=excluded, weight=1.0)
            ]
        )
    elif winner_name.startswith("return-avg") or winner_name.startswith("learned-w"):
        w_lgbm = mean_w if winner_name.startswith("learned-w") else 1.0
        w_ridge = (1 - mean_w) if winner_name.startswith("learned-w") else 1.0
        print(f"\nPersisting LightGBM+Ridge ({winner_name}) via run_training.")
        run_training(
            model_specs=[
                ModelSpec("lightgbm", params=persistable_lgbm_params, excluded_features=excluded, weight=w_lgbm),
                ModelSpec("ridge", params=dict(RIDGE_DEFAULT_PARAMS), excluded_features=excluded, weight=w_ridge),
            ]
        )
    else:
        print(f"\n{winner_name} beat production on a metric live scoring cannot serve; not persisting.")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
