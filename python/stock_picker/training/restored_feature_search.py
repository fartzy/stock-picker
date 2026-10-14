"""Compare restoring older prunes on matched forward dates, using all new inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_picker.features.catalog import list_feature_columns
from stock_picker.features.catalog_loader import sample_history
from stock_picker.log import get_logger
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.paths import data_root
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.feature_challenger import DEFAULT_OUTPUT_DIR
from stock_picker.training.feature_family_search import _evaluate, _load_pooled
from stock_picker.training.importance import highest_known_importance, importance_history_store, model_type_importance
from stock_picker.training.model import feature_columns, predict, train_lightgbm_rank
from stock_picker.training.rank_model import RANK_MODEL_NAME
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

RESTORED_FAMILIES = ("trend", "oscillators", "volume", "candle", "volatility")
logger = get_logger(__name__)


def restoration_variants(available: set[str], pruned: set[str], catalog: dict) -> dict[str, set[str]]:
    baseline = available - pruned
    variants = {"baseline": baseline}
    for family in RESTORED_FAMILIES:
        restored = set(catalog.get(family, ())) & pruned & available
        if restored:
            variants[f"restore_{family}"] = baseline | restored
    variants["restore_all"] = available
    return variants


def top_pruned_features(model, pruned: set[str], count: int) -> set[str]:
    gains = dict(zip(model.feature_names, model.estimator.feature_importance(importance_type="gain")))
    names = sorted((name for name in pruned if gains.get(name, 0) > 0), key=lambda name: (-gains[name], name))
    return set(names[:count])


def screen_restorations(train, catalog: dict, *, rounds: int = 100, folds: int = 3, threads: int = 2, selected=None, history=None) -> dict:
    if min(rounds, folds, threads) < 1:
        raise ValueError("rounds, folds and threads must be positive")
    start = max(UniverseStore().first_seen_by_ticker().values())
    recent = train[train["date"] >= start]
    if recent["date"].nunique() < (folds + 1) * 3:
        raise ValueError("not enough recent sessions for forward evaluation")
    pruned = PrunedFeatureStore().read()
    available = set(feature_columns(train))
    static = restoration_variants(available, pruned, catalog)
    names = list(static) + ["restore_top5", "restore_top10"]
    if selected is not None:
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError(f"unknown restoration variants: {sorted(unknown)}")
        names = [name for name in names if name in selected or name == "baseline"]
    results = {name: [] for name in names}
    for _, test_mask in walk_forward_splits(recent["date"], n_splits=folds):
        test = recent.loc[test_mask]
        fitting = train[train["date"] < test["date"].min()]
        logger.info("restoration fold: train through %s, test %s–%s", fitting["date"].max(), test["date"].min(), test["date"].max())
        all_model = train_lightgbm_rank(fitting, num_boost_round=rounds, params={"num_threads": threads}, included_features=available)
        dynamic = {
            "restore_top5": static["baseline"] | top_pruned_features(all_model, pruned, 5),
            "restore_top10": static["baseline"] | top_pruned_features(all_model, pruned, 10),
        }
        variants = {**static, **dynamic}
        observations = []
        for name in names:
            columns = variants[name]
            model = all_model if name == "restore_all" else train_lightgbm_rank(
                fitting, num_boost_round=rounds, params={"num_threads": threads}, included_features=columns,
            )
            metrics = _evaluate(test, predict(model, test), "rank")
            results[name].append({
                "train_through": str(fitting["date"].max()),
                "test_from": str(test["date"].min()), "test_through": str(test["date"].max()),
                "feature_count": len(model.feature_names),
                "restored": sorted(columns - static["baseline"]), **metrics,
            })
            observations.append((f"{name} through {str(fitting['date'].max())[:10]}", model_type_importance(model)))
            logger.info("%s: top20=%+.4f%%, Rank IC=%.4f", name, metrics["top20_mean_return"] * 100, metrics["rank_ic"])
        if history is not None:
            history.record_many(observations)
    return {
        "training_start": str(train["date"].min()), "train_through": str(train["date"].max()),
        "rows": len(train), "train_tickers": int(train["ticker"].nunique()),
        "rounds": rounds, "threads": threads, "folds": results,
    }


def save_restored_model(train, catalog: dict, *, variant: str, rounds: int, threads: int) -> dict:
    pruned = PrunedFeatureStore().read()
    available = set(feature_columns(train))
    variants = restoration_variants(available, pruned, catalog)
    params = {"num_threads": threads}
    if variant in {"restore_top5", "restore_top10"}:
        all_model = train_lightgbm_rank(train, num_boost_round=rounds, params=params, included_features=available)
        variants[variant] = variants["baseline"] | top_pruned_features(all_model, pruned, 5 if variant == "restore_top5" else 10)
    columns = variants[variant]
    name = f"rank_{variant}_full_{str(train['date'].max())[:10]}"
    store = ModelStore(data_dir=DEFAULT_OUTPUT_DIR)
    if store.exists(name):
        raise FileExistsError(name)
    model = train_lightgbm_rank(train, num_boost_round=rounds, params=params, included_features=columns)
    store.write(name, Ensemble(members=[model], weights=[1.0]))
    importance_history_store().record_many([(name, model_type_importance(model))])
    return {"artifact": str(DEFAULT_OUTPUT_DIR / f"{name}.pkl"), "restored": sorted(columns & pruned)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--variants", help="comma-separated comparison variants")
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--save-variant")
    parser.add_argument("--save-only", action="store_true")
    parser.add_argument("--check-holdout", action="store_true")
    args = parser.parse_args()
    if args.output_report and not args.output_report.is_absolute():
        args.output_report = data_root().parent / args.output_report
    if args.save_only and not args.save_variant:
        parser.error("--save-only requires --save-variant")
    if args.check_holdout and not args.save_variant:
        parser.error("--check-holdout requires --save-variant")
    if args.output_report and args.output_report.exists():
        raise FileExistsError(args.output_report)
    highest_known_importance()
    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train = _load_pooled([ticker for ticker in tickers if ticker not in holdout])
    catalog = list_feature_columns(sample_history())
    report = {} if args.save_only else screen_restorations(
        train, catalog, rounds=args.rounds, folds=args.folds, threads=args.threads,
        selected=args.variants.split(",") if args.variants else None,
        history=importance_history_store(),
    )
    if args.save_variant:
        report["saved_model"] = save_restored_model(train, catalog, variant=args.save_variant, rounds=args.rounds, threads=args.threads)
    if args.check_holdout:
        heldout = _load_pooled([ticker for ticker in tickers if ticker in holdout])
        heldout = heldout[heldout["date"] >= max(UniverseStore().first_seen_by_ticker().values())]
        path = Path(report["saved_model"]["artifact"])
        candidate = ModelStore(data_dir=path.parent).read(path.stem)
        baseline = ModelStore().read(RANK_MODEL_NAME)
        report["ticker_holdout"] = {
            "rows": len(heldout), "tickers": int(heldout["ticker"].nunique()),
            "baseline": _evaluate(heldout, predict(baseline.members[0], heldout), "rank"),
            "candidate": _evaluate(heldout, predict(candidate.members[0], heldout), "rank"),
        }
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
