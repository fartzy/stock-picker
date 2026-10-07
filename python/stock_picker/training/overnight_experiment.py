"""Controlled, row-level comparison of close-conditioned next-open features.

Every variant sees the same verified labels, prior-trained morning scores, and
chronological folds. This job writes research results, never the live artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.main import _load_pooled_dataset
from stock_picker.training.overnight_dataset import LABEL_COLUMN, PriceContract
from stock_picker.training.overnight_day_scores import generate_historical_day_scores
from stock_picker.training.overnight_job import build_verified_overnight_rows, _specs_from_saved_models
from stock_picker.training.overnight_model import (
    DEFAULT_ROUNDS,
    FIT_RESIDUAL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    _fit,
    _validated_training_frame,
    attach_historical_day_scores,
)
from stock_picker.training.rank_model import RANK_MODEL_NAME
from stock_picker.training.splits import walk_forward_splits


CORE_COLUMNS = ("assumed_day_return", "day_fit_predicted_return")
EXISTING_COLUMNS = tuple(
    column for column in MODEL_FEATURE_COLUMNS
    if column not in {"prior_return_5d", FIT_RESIDUAL_COLUMN}
)
WEIGHTED_SPLIT_GAIN = 3.0


@dataclass(frozen=True)
class Variant:
    name: str
    columns: tuple[str, ...]
    core_split_gain: float = 1.0

    def params(self) -> dict[str, object]:
        if self.core_split_gain == 1.0:
            return {}
        return {
            "feature_contri": [
                self.core_split_gain if column in CORE_COLUMNS else 1.0
                for column in self.columns
            ],
        }


VARIANTS = (
    Variant("core_2", CORE_COLUMNS),
    Variant("existing_13", EXISTING_COLUMNS),
    Variant("expanded_15", MODEL_FEATURE_COLUMNS),
    Variant("weighted_15", MODEL_FEATURE_COLUMNS, WEIGHTED_SPLIT_GAIN),
)
assert len(EXISTING_COLUMNS) == 13 and len(MODEL_FEATURE_COLUMNS) == 15


def evaluate_variants(
    frame: pd.DataFrame, *, n_splits: int = 2, rounds: int = DEFAULT_ROUNDS,
    params: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every held-out prediction and one comparable row per variant/fold."""
    if n_splits < 2:
        raise ValueError("at least two chronological test folds are required")
    rows = _validated_training_frame(frame)
    if rows["date"].nunique() < n_splits + 1:
        raise ValueError("not enough sessions for chronological evaluation")
    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for fold, (train_mask, test_mask) in enumerate(walk_forward_splits(rows["date"], n_splits), start=1):
        train, test = rows.loc[train_mask], rows.loc[test_mask]
        if train.empty or test.empty or train["date"].max() >= test["date"].min():
            raise ValueError("invalid chronological fold")
        actual_gap = test[LABEL_COLUMN].to_numpy(dtype=float)
        assumed_close = test["assumed_close"].to_numpy(dtype=float)
        ticker_means = train.groupby("ticker")[LABEL_COLUMN].mean()
        pooled_mean = float(train[LABEL_COLUMN].mean())
        historical_mean = test["ticker"].map(ticker_means).fillna(pooled_mean).to_numpy(dtype=float)
        for variant in VARIANTS:
            booster = _fit(train, {**(params or {}), **variant.params()}, rounds, variant.columns)
            predicted_gap = np.asarray(booster.predict(test[list(variant.columns)]), dtype=float)
            if not np.isfinite(predicted_gap).all():
                raise ValueError(f"{variant.name} returned a nonfinite prediction")
            block = test[["ticker", "date"]].copy().reset_index(drop=True)
            block.insert(0, "variant", variant.name)
            block.insert(1, "fold", fold)
            block["train_through"] = train["date"].max()
            block["assumed_close"] = assumed_close
            block["actual_gap"] = actual_gap
            block["predicted_gap"] = predicted_gap
            block["ticker_mean_gap"] = historical_mean
            block["actual_next_open"] = assumed_close * (1 + actual_gap)
            block["predicted_next_open"] = assumed_close * (1 + predicted_gap)
            block["unchanged_next_open"] = assumed_close
            block["model_abs_gap_error"] = np.abs(predicted_gap - actual_gap)
            block["zero_gap_abs_error"] = np.abs(actual_gap)
            block["ticker_mean_abs_error"] = np.abs(historical_mean - actual_gap)
            block["model_abs_open_error"] = np.abs(block["predicted_next_open"] - block["actual_next_open"])
            block["unchanged_abs_open_error"] = np.abs(assumed_close * actual_gap)
            predictions.append(block)
            nonzero = actual_gap != 0
            summaries.append({
                "variant": variant.name,
                "fold": fold,
                "train_through": train["date"].max().date().isoformat(),
                "test_start": test["date"].min().date().isoformat(),
                "test_end": test["date"].max().date().isoformat(),
                "rows": len(test),
                "model_gap_mae": float(block["model_abs_gap_error"].mean()),
                "zero_gap_mae": float(block["zero_gap_abs_error"].mean()),
                "ticker_mean_gap_mae": float(block["ticker_mean_abs_error"].mean()),
                "model_open_mae": float(block["model_abs_open_error"].mean()),
                "unchanged_open_mae": float(block["unchanged_abs_open_error"].mean()),
                "model_beats_zero_share": float((block["model_abs_gap_error"] < block["zero_gap_abs_error"]).mean()),
                "direction_accuracy": float(np.mean((predicted_gap[nonzero] > 0) == (actual_gap[nonzero] > 0)))
                if nonzero.any() else None,
                "up_base_rate": float(np.mean(actual_gap[nonzero] > 0)) if nonzero.any() else None,
            })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare overnight input sets on identical held-out dates")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--day-universe", choices=("all", "subset"), default="all")
    parser.add_argument("--fit-model", default="day_session_return")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start >= args.end or args.folds < 2 or len(args.tickers) != len(set(args.tickers)):
        parser.error("require an increasing date range, at least two folds, and unique tickers")
    if len(args.tickers) > 200:
        parser.error("provide no more than 200 explicit tickers")
    store = ModelStore()
    fit, rank = store.read(args.fit_model), store.read(RANK_MODEL_NAME)
    fit_specs, rank_spec = _specs_from_saved_models(fit, rank)
    day_tickers = UniverseStore().active_tickers() if args.day_universe == "all" else args.tickers
    day_training = _load_pooled_dataset(day_tickers, PriceStore(), FeatureStore())
    needed = set().union(*(set(member.feature_names) for member in (*fit.members, *rank.members)))
    for estimator in (getattr(fit, "stacked_svm_estimators", None) or {}).values():
        needed.update(estimator.feature_names)
    missing = needed - set(day_training) - set(STACKED_SVM_COLUMNS)
    if missing:
        raise ValueError(f"historical same-day inputs are missing: {sorted(missing)}")

    verified = {
        ticker: MassiveOvernightClient().fetch(ticker, args.start, args.end)
        for ticker in args.tickers
    }
    contract = PriceContract("massive", "raw", "massive_actions")
    labels, excluded = build_verified_overnight_rows(verified, contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    day_scores = generate_historical_day_scores(
        day_training, fit_specs=fit_specs, rank_spec=rank_spec,
        tracking_dir=args.output_dir / "mlruns", n_splits=args.folds,
    )
    scored_start = day_scores["date"].min()
    eligible = labels.merge(day_scores[["ticker", "date"]], on=["ticker", "date"], how="left", indicator=True)
    missing_scores = eligible["_merge"].eq("left_only")
    if (eligible.loc[missing_scores, "date"] >= scored_start).any():
        raise ValueError("missing non-warmup same-day scores")
    frame = attach_historical_day_scores(eligible.loc[~missing_scores].drop(columns="_merge"), day_scores)
    predictions, summaries = evaluate_variants(frame, n_splits=args.folds)
    predictions.to_csv(args.output_dir / "held_out_predictions.csv", index=False)
    summaries.to_csv(args.output_dir / "fold_summary.csv", index=False)
    metadata = {
        "tickers": args.tickers,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "day_universe": args.day_universe,
        "fit_model": args.fit_model,
        "label_rows": len(labels),
        "scored_rows": len(frame),
        "excluded_by_reason": excluded,
        "variants": [{"name": variant.name, "features": variant.columns,
                      "core_split_gain": variant.core_split_gain} for variant in VARIANTS],
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summaries.to_string(index=False))
    print(f"Wrote {len(predictions)} held-out variant predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
