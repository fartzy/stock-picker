"""Controlled, row-level comparison of close-conditioned next-open features.

Every variant sees the same verified labels, prior-trained morning scores, and
chronological folds. This job writes research results, never the live artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from stock_picker.training.overnight_cohort import (
    COHORT_SOURCE, COHORT_UNIVERSE_LIMIT, ranked_labeled_rows, select_ranked_cohort,
)
from stock_picker.training.overnight_dataset import LABEL_COLUMN, PriceContract, next_expected_session
from stock_picker.training.overnight_day_scores import generate_historical_day_scores
from stock_picker.training.overnight_job import (
    _specs_from_saved_models, build_verified_overnight_rows, fetch_start_with_prior_sessions,
)
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


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Illustrative sell-execution costs, not measured transaction costs.

    Half-spread and slippage are basis points of the relevant sale price; fee
    is dollars per share. The position is already owned, so no buy cost enters.
    """

    today_half_spread_bps: float = 5.0
    today_slippage_bps: float = 0.0
    today_fee_per_share: float = 0.0
    next_open_half_spread_bps: float = 5.0
    next_open_slippage_bps: float = 10.0
    next_open_fee_per_share: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(tuple(vars(self).values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("execution assumptions must be finite and nonnegative")

    def sell_today_cost(self, close: np.ndarray) -> np.ndarray:
        return close * (self.today_half_spread_bps + self.today_slippage_bps) / 10_000 + self.today_fee_per_share

    def sell_next_open_cost(self, opened: np.ndarray) -> np.ndarray:
        return opened * (self.next_open_half_spread_bps + self.next_open_slippage_bps) / 10_000 + self.next_open_fee_per_share


def evaluate_ranked_hold_policy(
    predictions: pd.DataFrame,
    verified_labels: pd.DataFrame,
    ranked_cohort: pd.DataFrame,
    assumptions: ExecutionAssumptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cost held-out Rank-pick predictions against selling at the actual close.

    All evaluated keys must be selected Rank names with verified raw labels.
    Equal-capital return is the mean of each edge divided by its actual close:
    one equal dollar allocation to each evaluated ticker-session, not a
    compounded portfolio return. Missing labels never promote lower ranks.
    """
    required = {"variant", "fold", "ticker", "date", "train_through", "assumed_close",
                "actual_gap", "actual_next_open", "predicted_next_open"}
    if required - set(predictions) or {"ticker", "date", "assumed_close", LABEL_COLUMN} - set(verified_labels):
        raise ValueError("cost evaluation needs held-out predictions and verified raw labels")
    if predictions.empty or predictions.duplicated(["variant", "ticker", "date"]).any():
        raise ValueError("each variant needs unique held-out ticker/date predictions")
    selected = ranked_labeled_rows(verified_labels, ranked_cohort)
    if selected.empty:
        raise ValueError("no verified Rank picks are available for cost evaluation")
    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="raise")
    rows["train_through"] = pd.to_datetime(rows["train_through"], errors="raise")
    if not (rows["train_through"] < rows["date"]).all():
        raise ValueError("cost evaluation requires strictly held-out predictions")
    checked = rows.merge(
        selected[["ticker", "date", "assumed_close", LABEL_COLUMN]],
        on=["ticker", "date"], how="left", validate="many_to_one", indicator=True,
        suffixes=("", "_verified"),
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError("cost evaluation contains a non-Rank or unverified row")
    close = checked["assumed_close"].to_numpy(dtype=float)
    actual_open = checked["actual_next_open"].to_numpy(dtype=float)
    predicted_open = checked["predicted_next_open"].to_numpy(dtype=float)
    verified_close = checked["assumed_close_verified"].to_numpy(dtype=float)
    actual_gap = checked["actual_gap"].to_numpy(dtype=float)
    verified_gap = checked[LABEL_COLUMN].to_numpy(dtype=float)
    if (not np.isfinite([close, actual_open, predicted_open, verified_close, actual_gap, verified_gap]).all()
            or (close <= 0).any() or (actual_open <= 0).any() or (predicted_open <= 0).any()
            or not np.allclose(close, verified_close, rtol=0, atol=1e-10)
            or not np.allclose(actual_gap, verified_gap, rtol=0, atol=1e-10)
            or not np.allclose(actual_open, close * (1 + verified_gap), rtol=0, atol=1e-8)):
        raise ValueError("cost evaluation prices do not match verified raw close/next-open labels")
    today_cost = assumptions.sell_today_cost(close)
    predicted_next_cost = assumptions.sell_next_open_cost(predicted_open)
    actual_next_cost = assumptions.sell_next_open_cost(actual_open)
    checked["sell_today_cost_per_share"] = today_cost
    checked["predicted_sell_next_open_cost_per_share"] = predicted_next_cost
    checked["actual_sell_next_open_cost_per_share"] = actual_next_cost
    checked["predicted_hold_edge_per_share"] = predicted_open - close + today_cost - predicted_next_cost
    checked["realized_hold_edge_per_share"] = actual_open - close + today_cost - actual_next_cost
    checked["hold_decision"] = checked["predicted_hold_edge_per_share"] > 0
    checked["policy_edge_per_share"] = np.where(checked["hold_decision"], checked["realized_hold_edge_per_share"], 0.0)
    checked["policy_equal_capital_return"] = checked["policy_edge_per_share"] / close
    checked["always_hold_equal_capital_return"] = checked["realized_hold_edge_per_share"] / close

    def summarize(variant: str, fold: int | str, group: pd.DataFrame) -> dict[str, object]:
        return {
            "variant": variant, "fold": fold, "rows": len(group),
            "hold_share": float(group["hold_decision"].mean()),
            "policy_edge_per_share": float(group["policy_edge_per_share"].mean()),
            "always_sell_edge_per_share": 0.0,
            "always_hold_edge_per_share": float(group["realized_hold_edge_per_share"].mean()),
            "policy_equal_capital_return": float(group["policy_equal_capital_return"].mean()),
            "always_sell_equal_capital_return": 0.0,
            "always_hold_equal_capital_return": float(group["always_hold_equal_capital_return"].mean()),
        }

    summaries = [
        summarize(variant, int(fold), group)
        for (variant, fold), group in checked.groupby(["variant", "fold"], sort=True)
    ]
    summaries.extend(
        summarize(variant, "all", group)
        for variant, group in checked.groupby("variant", sort=True)
    )
    return checked.drop(columns=["_merge", "assumed_close_verified", LABEL_COLUMN]), pd.DataFrame(summaries)


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


def fetch_verified_experiment_rows(
    tickers: list[str], start: date, end: date,
    client: MassiveOvernightClient, contract: PriceContract,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch six prior closes, then keep labels only in requested [start, end)."""
    if start >= end:
        raise ValueError("experiment start must precede end")
    fetch_start = fetch_start_with_prior_sessions(start)
    verified = {ticker: client.fetch(ticker, fetch_start, end) for ticker in tickers}
    labels, excluded = build_verified_overnight_rows(verified, contract)
    dates = pd.to_datetime(labels["date"], errors="raise")
    requested = labels.loc[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))].copy()
    if requested.empty:
        raise ValueError("no verified raw close/next-open labels in requested date window")
    return requested.reset_index(drop=True), excluded


def labelable_ranked_cohort(cohort: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Selections whose next XNYS open can exist in the fetched date range."""
    window = cohort.loc[
        (cohort["date"] >= pd.Timestamp(start)) & (cohort["date"] < pd.Timestamp(end))
    ]
    labelable_dates = {
        session for session in window["date"].drop_duplicates()
        if (following := next_expected_session(session.date())) is not None and following <= end
    }
    return window.loc[window["date"].isin(labelable_dates)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare overnight input sets on identical held-out dates")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--day-universe", choices=("all", "subset"), default="all")
    parser.add_argument("--fit-model", default="day_session_return")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranked-cost-eval", action="store_true",
                        help="also evaluate held-out hold decisions on verified Rank top-20 picks")
    parser.add_argument("--today-half-spread-bps", type=float, default=5.0)
    parser.add_argument("--today-slippage-bps", type=float, default=0.0)
    parser.add_argument("--today-fee-per-share", type=float, default=0.0)
    parser.add_argument("--next-open-half-spread-bps", type=float, default=5.0)
    parser.add_argument("--next-open-slippage-bps", type=float, default=10.0)
    parser.add_argument("--next-open-fee-per-share", type=float, default=0.0)
    args = parser.parse_args()
    if args.start >= args.end or args.folds < 2 or len(args.tickers) != len(set(args.tickers)):
        parser.error("require an increasing date range, at least two folds, and unique tickers")
    if len(args.tickers) > 200:
        parser.error("provide no more than 200 explicit tickers")
    if args.ranked_cost_eval and args.day_universe != "all":
        parser.error("ranked cost evaluation requires the full same-day universe")
    try:
        assumptions = ExecutionAssumptions(
            today_half_spread_bps=args.today_half_spread_bps,
            today_slippage_bps=args.today_slippage_bps,
            today_fee_per_share=args.today_fee_per_share,
            next_open_half_spread_bps=args.next_open_half_spread_bps,
            next_open_slippage_bps=args.next_open_slippage_bps,
            next_open_fee_per_share=args.next_open_fee_per_share,
        )
    except ValueError as exc:
        parser.error(str(exc))
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

    contract = PriceContract("massive", "raw", "massive_actions")
    labels, excluded = fetch_verified_experiment_rows(
        args.tickers, args.start, args.end, MassiveOvernightClient(), contract,
    )
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
        "exclusion_scope": "all fetched provider bars, including warmup and final unlabeled bar",
        "variants": [{"name": variant.name, "features": variant.columns,
                      "core_split_gain": variant.core_split_gain} for variant in VARIANTS],
    }
    print(summaries.to_string(index=False))
    if args.ranked_cost_eval:
        cohort = select_ranked_cohort(day_scores, day_training)
        ranked_labels = ranked_labeled_rows(labels, cohort)
        if ranked_labels.empty:
            raise ValueError("no verified raw close/next-open labels overlap dated Rank top-20 picks")
        ranked_frame = attach_historical_day_scores(ranked_labels, day_scores)
        ranked_predictions, _ = evaluate_variants(ranked_frame, n_splits=args.folds)
        costed_rows, costed_summary = evaluate_ranked_hold_policy(
            ranked_predictions, labels, cohort, assumptions,
        )
        label_window_cohort = labelable_ranked_cohort(cohort, args.start, args.end)
        requested_cohort = label_window_cohort.loc[label_window_cohort["ticker"].isin(args.tickers)]
        cohort_scope = "requested-ticker intersection of reconstructed Rank top 20"
        coverage_denominator = "Rank selections with next XNYS session at or before fetched end date"
        costed_summary["cohort_scope"] = cohort_scope
        costed_summary["coverage_denominator"] = coverage_denominator
        costed_summary["selected_ticker_dates_in_label_window"] = len(label_window_cohort)
        costed_summary["selected_requested_ticker_dates"] = len(requested_cohort)
        costed_summary["selected_verified_label_rows"] = len(ranked_labels)
        for name, value in asdict(assumptions).items():
            costed_summary[name] = value
        costed_rows.to_csv(args.output_dir / "held_out_ranked_costed_predictions.csv", index=False)
        costed_summary.to_csv(args.output_dir / "ranked_hold_policy_summary.csv", index=False)
        metadata["ranked_cost_evaluation"] = {
            "cohort_source": COHORT_SOURCE,
            "cohort_universe_limit": COHORT_UNIVERSE_LIMIT,
            "rank_top_k": 20,
            "cohort_scope": cohort_scope,
            "selected_ticker_dates_in_label_window": len(label_window_cohort),
            "coverage_denominator": coverage_denominator,
            "selected_requested_ticker_dates": len(requested_cohort),
            "selected_verified_label_rows": len(ranked_labels),
            "full_rank_ticker_coverage": len(requested_cohort) == len(label_window_cohort),
            "full_rank_verified_label_coverage": len(ranked_labels) == len(label_window_cohort),
            "held_out_ticker_dates_per_variant": len(costed_rows) // len(VARIANTS),
            "execution_assumptions": asdict(assumptions),
            "cost_assumption_status": "illustrative scenario, not measured execution",
            "equal_capital_method": "arithmetic mean of per-row incremental edge / actual raw close",
            "decision_timing": "after actual close; not an intraday strategy backtest",
        }
        print(f"Rank cost cohort: {len(ranked_labels)} verified requested-ticker rows "
              f"of {len(label_window_cohort)} selected top-20 ticker-dates in the label window")
        print(costed_summary.to_string(index=False))
    (args.output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(predictions)} held-out variant predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
