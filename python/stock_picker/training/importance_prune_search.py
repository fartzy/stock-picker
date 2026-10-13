"""Forward-test small gain-importance prunes of the full-history Rank model.

Feature selection is refit inside each chronological training fold. The
research artifact, if requested, is separate from the live morning model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_picker.features.pruning import RESEARCH_CANDIDATE_COLUMNS
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.feature_challenger import DEFAULT_OUTPUT_DIR
from stock_picker.training.feature_family_search import _evaluate, _load_pooled
from stock_picker.training.model import feature_columns, predict, train_lightgbm_rank
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

PRUNE_COUNTS = (0, 3, 5)


def lowest_gain_features(model, count: int) -> list[dict]:
    """Return deterministic bottom-gain names from one training-only model."""
    if count < 0 or count >= len(model.feature_names):
        raise ValueError("prune count must leave at least one feature")
    gains = model.estimator.feature_importance(importance_type="gain")
    if len(gains) != len(model.feature_names):
        raise ValueError("feature importance does not match the trained feature list")
    ranked = sorted(zip(model.feature_names, gains), key=lambda item: (item[1], item[0]))
    return [
        {"name": name, "gain": float(gain), "new_feature": name in RESEARCH_CANDIDATE_COLUMNS}
        for name, gain in ranked[:count]
    ]


def compare_prunes(train, *, rounds: int = 100, folds: int = 3) -> dict:
    """Compare all, minus 3, and minus 5 on identical future dates."""
    if rounds < 1 or folds < 1:
        raise ValueError("rounds and folds must be positive")
    first_seen = UniverseStore().first_seen_by_ticker()
    if not first_seen:
        raise ValueError("universe first-seen dates are required for point-in-time features")
    recent = train[train["date"] >= max(first_seen.values())].copy()
    if recent["date"].nunique() < (folds + 1) * 3:
        raise ValueError("not enough post-registry sessions for forward evaluation")
    excluded = PrunedFeatureStore().read()
    columns = set(feature_columns(train, excluded))
    if not RESEARCH_CANDIDATE_COLUMNS <= columns:
        raise ValueError("all 19 new research features are required")
    if any(recent[name].notna().sum() < len(recent) // 2 for name in RESEARCH_CANDIDATE_COLUMNS):
        raise ValueError("the new features lack adequate recent-date coverage")

    results = {str(count): [] for count in PRUNE_COUNTS}
    for _, test_mask in walk_forward_splits(recent["date"], n_splits=folds):
        test = recent.loc[test_mask]
        fitting = train[train["date"] < test["date"].min()]
        full_model = train_lightgbm_rank(
            fitting, num_boost_round=rounds,
            excluded_features=excluded, included_features=columns,
        )
        bottom = lowest_gain_features(full_model, max(PRUNE_COUNTS))
        for count in PRUNE_COUNTS:
            dropped = bottom[:count]
            model = full_model if count == 0 else train_lightgbm_rank(
                fitting, num_boost_round=rounds,
                excluded_features=excluded,
                included_features=columns - {item["name"] for item in dropped},
            )
            results[str(count)].append({
                "train_through": str(fitting["date"].max()),
                "test_from": str(test["date"].min()),
                "test_through": str(test["date"].max()),
                "feature_count": len(model.feature_names),
                "dropped": dropped,
                **_evaluate(test, predict(model, test), "rank"),
            })
    return {
        "training_start": str(train["date"].min()),
        "evaluation_start": str(recent["date"].min()),
        "train_through": str(train["date"].max()),
        "train_rows": len(train),
        "train_tickers": int(train["ticker"].nunique()),
        "rounds": rounds,
        "full_feature_count": len(columns),
        "folds": results,
    }


def train_pruned_research_model(
    train, *, count: int, rounds: int = 100, output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Fit one full-history research model; never replace the live artifact."""
    if count not in PRUNE_COUNTS[1:]:
        raise ValueError("choose one tested nonzero prune count")
    name = f"rank_gain_prune_{count}_full_{str(train['date'].max())[:10]}"
    destination = Path(output_dir) / f"{name}.pkl"
    if destination.exists():
        raise FileExistsError(f"research artifact already exists: {destination}")
    excluded = PrunedFeatureStore().read()
    columns = set(feature_columns(train, excluded))
    if not RESEARCH_CANDIDATE_COLUMNS <= columns:
        raise ValueError("all 19 new research features are required")
    full_model = train_lightgbm_rank(
        train, num_boost_round=rounds,
        excluded_features=excluded, included_features=columns,
    )
    dropped = lowest_gain_features(full_model, count)
    pruned_model = train_lightgbm_rank(
        train, num_boost_round=rounds,
        excluded_features=excluded,
        included_features=columns - {item["name"] for item in dropped},
    )
    ModelStore(data_dir=output_dir).write(name, Ensemble(members=[pruned_model], weights=[1.0]))
    return {"artifact": str(destination), "feature_count": len(pruned_model.feature_names), "dropped": dropped}


def compare_saved_holdout(holdout, *, through: str, count: int, model_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Diagnostic only: held-out tickers were used by earlier model research."""
    start = max(UniverseStore().first_seen_by_ticker().values())
    recent = holdout[holdout["date"] >= start]
    if recent.empty:
        raise ValueError("no post-registry held-out ticker rows")
    store = ModelStore(data_dir=model_dir)
    results = {}
    for label, name in (
        ("all", f"rank_all_full_{through}"),
        (f"prune_{count}", f"rank_gain_prune_{count}_full_{through}"),
    ):
        ensemble = store.read(name)
        results[label] = _evaluate(recent, predict(ensemble.members[0], recent), "rank")
    return {
        "rows": len(recent),
        "tickers": int(recent["ticker"].nunique()),
        "metrics": results,
        "caveat": "these ticker holdouts were examined in earlier feature research",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--save-pruned", type=int, choices=PRUNE_COUNTS[1:])
    parser.add_argument("--save-only", action="store_true", help="skip repeated folds when saving a previously evaluated count")
    parser.add_argument("--evaluate-saved", type=int, choices=PRUNE_COUNTS[1:])
    parser.add_argument("--output-model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.save_only and args.save_pruned is None:
        parser.error("--save-only requires --save-pruned")
    if args.evaluate_saved is not None and args.save_pruned is not None:
        parser.error("--evaluate-saved and --save-pruned are separate steps")
    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train = _load_pooled([ticker for ticker in tickers if ticker not in holdout])
    report = {} if args.save_only or args.evaluate_saved is not None else compare_prunes(train, rounds=args.rounds, folds=args.folds)
    if args.save_pruned is not None:
        report["research_model"] = train_pruned_research_model(
            train, count=args.save_pruned, rounds=args.rounds, output_dir=args.output_model_dir,
        )
    if args.evaluate_saved is not None:
        heldout = _load_pooled([ticker for ticker in tickers if ticker in holdout])
        report["ticker_holdout"] = compare_saved_holdout(
            heldout, through=str(train["date"].max())[:10],
            count=args.evaluate_saved, model_dir=args.output_model_dir,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
