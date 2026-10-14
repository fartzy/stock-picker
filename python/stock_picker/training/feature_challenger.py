"""Train an isolated Rank challenger without replacing the live model.

The candidate is compared with an otherwise identical baseline on paired
chronological folds and the deterministic ticker holdout. ``--scope full``
trains on all history while testing only post-registry sessions where
point-in-time breadth is available; older breadth/path values remain missing.
The resulting artifact is written under ``data/models/research/`` and is not
selected by serving or the morning training job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_picker.features.breadth import BREADTH_COLUMNS
from stock_picker.features.pruning import RESEARCH_CANDIDATE_COLUMNS
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.paths import data_root
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.feature_family_search import _evaluate, _load_pooled, variants
from stock_picker.training.model import feature_columns, predict, train_lightgbm_rank
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

CHALLENGER_VARIANTS = {
    "breadth_price": frozenset(BREADTH_COLUMNS[:3]),
    "all": RESEARCH_CANDIDATE_COLUMNS,
}
DEFAULT_OUTPUT_DIR = data_root() / "models" / "research"


def paired_rank_evaluation(
    train, holdout, *, variant: str = "breadth_price", scope: str = "recent",
    rounds: int, folds: int,
) -> tuple[dict, set[str]]:
    """Evaluate one candidate on matched post-registry dates, without writes."""
    if variant not in CHALLENGER_VARIANTS:
        raise ValueError(f"unknown challenger variant: {variant}")
    if scope not in {"recent", "full"}:
        raise ValueError(f"unknown training scope: {scope}")
    if rounds < 1 or folds < 1:
        raise ValueError("rounds and folds must be positive")
    first_seen = UniverseStore().first_seen_by_ticker()
    if not first_seen:
        raise ValueError("universe first-seen dates are required for point-in-time breadth")
    recent_start = max(first_seen.values())
    recent = train[train["date"] >= recent_start].copy()
    if scope == "recent":
        train = recent
    holdout = holdout[holdout["date"] >= recent_start].copy()
    if recent["date"].nunique() < (folds + 1) * 3 or holdout.empty:
        raise ValueError("not enough recent sessions or held-out rows for challenger evaluation")

    # Bypass only the *research* gate. Explicit user quality prunes still apply.
    excluded = PrunedFeatureStore().read()
    selected = variants(feature_columns(train, excluded), "recent")
    baseline, candidate = selected["baseline"], selected[variant]
    added = candidate - baseline
    if added != CHALLENGER_VARIANTS[variant]:
        raise ValueError(f"all unpruned {variant} features are required")
    if any(recent[name].notna().sum() < len(recent) // 2 for name in added):
        raise ValueError("the candidate lacks adequate recent-date coverage")

    paired = {"baseline": [], variant: []}
    for _, test_mask in walk_forward_splits(recent["date"], n_splits=folds):
        test_dates = recent.loc[test_mask, "date"]
        fold_train = train[train["date"] < test_dates.min()]
        fold_test = recent.loc[test_mask]
        for name, columns in (("baseline", baseline), (variant, candidate)):
            model = train_lightgbm_rank(
                fold_train, num_boost_round=rounds,
                excluded_features=excluded, included_features=columns,
            )
            paired[name].append({
                "train_through": str(fold_train["date"].max()),
                "test_from": str(fold_test["date"].min()),
                "test_through": str(fold_test["date"].max()),
                **_evaluate(fold_test, predict(model, fold_test), "rank"),
            })

    holdout_metrics = {}
    for name, columns in (("baseline", baseline), (variant, candidate)):
        model = train_lightgbm_rank(
            train, num_boost_round=rounds,
            excluded_features=excluded, included_features=columns,
        )
        holdout_metrics[name] = _evaluate(holdout, predict(model, holdout), "rank")

    return {
        "variant": variant,
        "scope": scope,
        "start": str(train["date"].min()),
        "evaluation_start": str(recent["date"].min()),
        "train_through": str(train["date"].max()),
        "train_rows": len(train),
        "train_tickers": int(train["ticker"].nunique()),
        "holdout_rows": len(holdout),
        "holdout_tickers": int(holdout["ticker"].nunique()),
        "rounds": rounds,
        "folds": paired,
        "ticker_holdout": holdout_metrics,
        "new_features": sorted(added),
    }, candidate


def train_and_save_challenger(
    train, holdout, *, variant: str = "breadth_price", scope: str = "recent",
    rounds: int = 100, folds: int = 3,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Persist one named research artifact only after evaluation succeeds."""
    result, columns = paired_rank_evaluation(
        train, holdout, variant=variant, scope=scope, rounds=rounds, folds=folds,
    )
    training_frame = train[train["date"] >= result["start"]].copy()
    scope_suffix = "_full" if scope == "full" else ""
    name = f"rank_{variant}{scope_suffix}_{result['train_through'][:10]}"
    output_path = Path(output_dir) / f"{name}.pkl"
    if output_path.exists():
        raise FileExistsError(f"research artifact already exists: {output_path}")
    model = train_lightgbm_rank(
        training_frame, num_boost_round=rounds,
        excluded_features=PrunedFeatureStore().read(), included_features=columns,
    )
    ModelStore(data_dir=output_dir).write(name, Ensemble(members=[model], weights=[1.0]))
    return {**result, "artifact": str(output_path), "feature_count": len(model.feature_names)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(CHALLENGER_VARIANTS), default="breadth_price")
    parser.add_argument("--scope", choices=("recent", "full"), default="recent")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output-model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train = _load_pooled([ticker for ticker in tickers if ticker not in holdout])
    heldout = _load_pooled([ticker for ticker in tickers if ticker in holdout])
    if args.evaluate_only:
        report, _ = paired_rank_evaluation(
            train, heldout, variant=args.variant, scope=args.scope,
            rounds=args.rounds, folds=args.folds,
        )
    else:
        report = train_and_save_challenger(
            train, heldout, variant=args.variant, scope=args.scope,
            rounds=args.rounds, folds=args.folds,
            output_dir=args.output_model_dir,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
