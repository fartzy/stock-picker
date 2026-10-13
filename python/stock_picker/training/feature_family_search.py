"""Read-only walk-forward ablation of the new price-pattern feature families.

This is a fast single-model screen, not the production Fit ensemble or a model
promotion. It never trains on the deterministic ticker holdout and never writes
model, pruning, or training configuration files.

Run full-history volatility screen:
    bazelisk run //python/stock_picker/training:feature_family_search -- --scope full
Run recent point-in-time breadth/path screen:
    bazelisk run //python/stock_picker/training:feature_family_search -- --scope recent
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from stock_picker.features.breadth import BREADTH_COLUMNS
from stock_picker.features.pruning import RESEARCH_CANDIDATE_COLUMNS
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.features.trajectory import PATH_COLUMNS
from stock_picker.features.volatility import SEMIVOL_WINDOWS, SHORT_VOLATILITY_LAGS
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import rank_ic, simulate_top_k
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.model import (
    feature_columns,
    predict,
    train_lightgbm,
    train_lightgbm_rank,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

VOLATILITY_SHAPE_COLUMNS = frozenset({
    "volatility_32d",
    "absolute_return_2d",
    *(f"volatility_5d_lag_{lag}d" for lag in SHORT_VOLATILITY_LAGS),
})
VOLATILITY_DIRECTION_COLUMNS = frozenset({
    *(f"{side}_semivol_{window}d" for side in ("upside", "downside") for window in SEMIVOL_WINDOWS),
    *(f"semivol_balance_{window}d" for window in SEMIVOL_WINDOWS),
})
VOLATILITY_COLUMNS = VOLATILITY_SHAPE_COLUMNS | VOLATILITY_DIRECTION_COLUMNS
FAMILIES = {
    "volatility": VOLATILITY_COLUMNS,
    "breadth": frozenset(BREADTH_COLUMNS),
    "trajectory": frozenset(PATH_COLUMNS),
}
SUBFAMILIES = {
    "volatility_shape": VOLATILITY_SHAPE_COLUMNS,
    "volatility_direction": VOLATILITY_DIRECTION_COLUMNS,
    "breadth_price": frozenset(BREADTH_COLUMNS[:3]),
    "breadth_volume": frozenset(BREADTH_COLUMNS[3:]),
    "trajectory_shape": frozenset(PATH_COLUMNS[:2]),
    "trajectory_peer": frozenset(PATH_COLUMNS[2:]),
}
ALL_CANDIDATES = frozenset().union(*FAMILIES.values())
assert ALL_CANDIDATES == RESEARCH_CANDIDATE_COLUMNS


def variants(columns: list[str], scope: str) -> dict[str, set[str]]:
    """Keep the baseline identical across family ablations."""
    available = set(columns)
    baseline = available - ALL_CANDIDATES
    names = (
        ("volatility", "volatility_shape", "volatility_direction")
        if scope == "full"
        else (*FAMILIES, "breadth_price", "breadth_volume", "trajectory_shape", "trajectory_peer")
    )
    result = {"baseline": baseline}
    groups = {**FAMILIES, **SUBFAMILIES}
    result.update({name: baseline | (groups[name] & available) for name in names})
    if scope == "recent":
        result["all"] = baseline | (ALL_CANDIDATES & available)
    return result


def _load_pooled(tickers: list[str]) -> pd.DataFrame:
    prices, features = PriceStore(), FeatureStore()
    histories, features_by_ticker = {}, {}
    for ticker in tickers:
        try:
            histories[ticker] = prices.read(ticker)
            features_by_ticker[ticker] = features.read(ticker)
        except FileNotFoundError:
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def _evaluate(test: pd.DataFrame, predicted: np.ndarray, model_type: str) -> dict[str, float]:
    prediction = pd.Series(predicted, index=test.index)
    actual = test[LABEL_COLUMN]
    top_five = simulate_top_k(prediction, actual, test["date"], k=5)
    metrics = {
        "rank_ic": rank_ic(prediction, actual, test["date"]),
        "top5_mean_return": float(top_five["avg_return"]),
        "top5_hit_rate": float(top_five["hit_rate"]),
    }
    if model_type == "fit":
        metrics["mae"] = float(np.mean(np.abs(predicted - actual.to_numpy())))
        metrics["directional_accuracy"] = float(
            np.mean(np.sign(predicted) == np.sign(actual.to_numpy()))
        )
    return metrics


def screen(
    pooled: pd.DataFrame,
    scope: str,
    rounds: int = 40,
    n_splits: int = 3,
) -> dict:
    """Compare paired dates/folds; never use the ticker holdout for selection."""
    # The screen may opt into provisional candidates, but must still honor
    # explicit quality prunes recorded by the user.
    excluded = PrunedFeatureStore().read()
    available = feature_columns(pooled, excluded)
    candidates = variants(available, scope)
    if scope == "recent":
        first_seen = UniverseStore().first_seen_by_ticker()
        start = max(first_seen.values())
        pooled = pooled[pooled["date"] >= start].copy()
    else:
        start = str(pooled["date"].min())
    if pooled["date"].nunique() < (n_splits + 1) * 3:
        raise ValueError("not enough distinct sessions for the requested walk-forward screen")
    splits = walk_forward_splits(pooled["date"], n_splits=n_splits)
    model_types = ("fit",) if scope == "full" else ("fit", "rank")
    results = {}
    for model_type in model_types:
        results[model_type] = {}
        for name, selected in candidates.items():
            folds = []
            for train_mask, test_mask in splits:
                train, test = pooled[train_mask], pooled[test_mask]
                trainer = train_lightgbm if model_type == "fit" else train_lightgbm_rank
                model = trainer(
                    train,
                    num_boost_round=rounds,
                    excluded_features=excluded,
                    included_features=selected,
                )
                folds.append({
                    "train_through": str(train["date"].max()),
                    "test_from": str(test["date"].min()),
                    "test_through": str(test["date"].max()),
                    "test_rows": len(test),
                    **_evaluate(test, predict(model, test), model_type),
                })
            results[model_type][name] = folds
    coverage = {
        name: float(pooled[list(columns)].notna().mean().mean())
        for name, columns in FAMILIES.items()
        if columns <= set(pooled.columns)
    }
    return {
        "scope": scope,
        "start": start,
        "end": str(pooled["date"].max()),
        "rows": len(pooled),
        "days": int(pooled["date"].nunique()),
        "rounds": rounds,
        "folds": n_splits,
        "candidate_coverage": coverage,
        "results": results,
    }


def check_ticker_holdout(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    variant: str,
    rounds: int = 40,
) -> dict:
    """One preselected candidate versus baseline on never-trained tickers."""
    if variant not in {**FAMILIES, **SUBFAMILIES}:
        raise ValueError(f"unknown holdout variant: {variant}")
    start = max(UniverseStore().first_seen_by_ticker().values())
    train = train[train["date"] >= start]
    holdout = holdout[holdout["date"] >= start]
    excluded = PrunedFeatureStore().read()
    selected = variants(feature_columns(train, excluded), "recent")
    if variant not in selected:
        selected[variant] = selected["baseline"] | (
            {**FAMILIES, **SUBFAMILIES}[variant] & set(train.columns)
        )
    results = {}
    for name in ("baseline", variant):
        model = train_lightgbm_rank(
            train,
            num_boost_round=rounds,
            excluded_features=excluded,
            included_features=selected[name],
        )
        results[name] = _evaluate(holdout, predict(model, holdout), "rank")
    return {
        "variant": variant,
        "train_tickers": int(train["ticker"].nunique()),
        "holdout_tickers": int(holdout["ticker"].nunique()),
        "holdout_rows": len(holdout),
        "days": int(holdout["date"].nunique()),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("full", "recent"), required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--holdout-variant", choices=tuple({**FAMILIES, **SUBFAMILIES}))
    args = parser.parse_args()
    universe = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(universe)
    pooled = _load_pooled([ticker for ticker in universe if ticker not in holdout])
    if args.holdout_variant:
        heldout = _load_pooled([ticker for ticker in universe if ticker in holdout])
        result = check_ticker_holdout(pooled, heldout, args.holdout_variant, args.rounds)
    else:
        result = screen(pooled, args.scope, args.rounds, args.folds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
