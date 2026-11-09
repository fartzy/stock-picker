"""Walk-forward lambdarank vs production regression LightGBM.

Ranking is not "fit winners." Regression minimizes error on the return
number. Lambdarank only tries to put higher-return names above lower-return
names *that same day*. Output is a relative score, not a percent -- live
0.5% gating does not apply. This script measures Rank IC and top-K hit
rate. It does not overwrite latest unless ranking wins both walk-forward
Rank IC and holdout Rank IC.
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
from stock_picker.training.backtest import rank_ic
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.main import _load_pooled_dataset
from stock_picker.training.model import (
    LIGHTGBM_DEFAULT_PARAMS,
    predict,
    train_lightgbm,
    train_lightgbm_rank,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

print = functools.partial(print, flush=True)

N_SPLITS = 4
N_CORES = os.cpu_count() or 4
TOP_KS = (5, 20)


def _lgbm_params() -> dict:
    return {**LIGHTGBM_DEFAULT_PARAMS, "num_threads": N_CORES}


def _top_k_hit(pred: np.ndarray, actual: np.ndarray, dates: pd.Series, k: int) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual, "date": dates})
    hits = []
    for _, day in frame.groupby("date"):
        if len(day) < k:
            continue
        top = day.nlargest(k, "pred")
        hits.append(float((top["actual"] > 0).mean()))
    return float(np.mean(hits)) if hits else float("nan")


def _top_k_mean_return(pred: np.ndarray, actual: np.ndarray, dates: pd.Series, k: int) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual, "date": dates})
    means = []
    for _, day in frame.groupby("date"):
        if len(day) < k:
            continue
        means.append(float(day.nlargest(k, "pred")["actual"].mean()))
    return float(np.mean(means)) if means else float("nan")


def _report(name: str, pred: np.ndarray, actual: np.ndarray, dates: pd.Series) -> dict:
    ic = rank_ic(pd.Series(pred), pd.Series(actual), dates)
    row = {"name": name, "rank_ic": ic}
    print(f"  {name}: Rank IC={ic:.4f}", end="")
    for k in TOP_KS:
        hit = _top_k_hit(pred, actual, dates, k)
        mean_ret = _top_k_mean_return(pred, actual, dates, k)
        row[f"top{k}_hit"] = hit
        row[f"top{k}_mean"] = mean_ret
        print(f"  top-{k} hit={hit:.4f} mean={mean_ret * 100:.3f}%", end="")
    print()
    return row


def main() -> None:
    t0 = time.time()
    excluded = pruned_features()
    tickers = UniverseStore().active_tickers()
    holdout_set = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout_set]
    holdout_tickers = [t for t in tickers if t in holdout_set]
    price_store, feature_store = PriceStore(), FeatureStore()
    params = _lgbm_params()

    print(f"loading train pool ({len(train_tickers)} tickers)...")
    pooled_train = _load_pooled_dataset(train_tickers, price_store, feature_store)
    print(f"train rows={len(pooled_train)} ({time.time() - t0:.1f}s)")

    print("\n=== Walk-forward (train tickers, future dates) ===")
    splits = walk_forward_splits(pooled_train["date"], n_splits=N_SPLITS)
    reg_ics, rank_ics = [], []
    for i, (train_mask, test_mask) in enumerate(splits, start=1):
        started = time.time()
        train_frame = pooled_train[train_mask]
        test_frame = pooled_train[test_mask]
        regression = train_lightgbm(train_frame, params=params, excluded_features=excluded)
        ranking = train_lightgbm_rank(train_frame, params=params, excluded_features=excluded)
        actual = test_frame[LABEL_COLUMN].to_numpy()
        dates = test_frame["date"]
        print(f"fold {i} ({time.time() - started:.1f}s)")
        reg = _report("  regression", predict(regression, test_frame), actual, dates)
        ran = _report("  lambdarank", predict(ranking, test_frame), actual, dates)
        reg_ics.append(reg["rank_ic"])
        rank_ics.append(ran["rank_ic"])
    print(
        f"\nWalk-forward mean Rank IC  regression={np.mean(reg_ics):.4f}  "
        f"lambdarank={np.mean(rank_ics):.4f}"
    )

    print("\n=== Holdout (never-seen tickers) ===")
    pooled_holdout = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
    regression = train_lightgbm(pooled_train, params=params, excluded_features=excluded)
    ranking = train_lightgbm_rank(pooled_train, params=params, excluded_features=excluded)
    actual = pooled_holdout[LABEL_COLUMN].to_numpy()
    dates = pooled_holdout["date"]
    hold_reg = _report("regression", predict(regression, pooled_holdout), actual, dates)
    hold_rank = _report("lambdarank", predict(ranking, pooled_holdout), actual, dates)

    wf_win = np.mean(rank_ics) > np.mean(reg_ics)
    ho_win = hold_rank["rank_ic"] > hold_reg["rank_ic"]
    print(f"\nLambdarank wins walk-forward Rank IC: {wf_win}; wins holdout Rank IC: {ho_win}")
    if wf_win and ho_win:
        print("Worth a top-K serving path -- not swapping latest (scores are not percents).")
    else:
        print("Not replacing production LightGBM -- ranking did not beat regression on Rank IC.")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
