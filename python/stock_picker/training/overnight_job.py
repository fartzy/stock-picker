"""One bounded, explicit training job for the separate next-open model.

The caller supplies verified Massive bars and a same-day training frame.
The job never falls back to legacy OHLCV data for the overnight label and
does not overwrite the open-to-close Fit or Rank artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import pandas as pd
import exchange_calendars as xcals

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, VerifiedOvernightBars
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.paths import data_root
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.ensemble import Ensemble, ModelSpec
from stock_picker.training.main import MODEL_NAME as DAY_MODEL_NAME, _load_pooled_dataset
from stock_picker.training.overnight_cohort import (
    COHORT_SOURCE, COHORT_UNIVERSE_LIMIT, make_ranked_cohort_manifest,
    ranked_labeled_rows, save_ranked_cohort_manifest, select_ranked_cohort,
)
from stock_picker.training.overnight_dataset import PRIOR_CLOSE_COUNT, PriceContract, build_overnight_training_frame
from stock_picker.training.overnight_day_scores import generate_historical_day_scores
from stock_picker.training.overnight_model import (
    MODEL_NAME,
    OvernightModel,
    attach_historical_day_scores,
    train_overnight_model,
)
from stock_picker.training.rank_model import RANK_TOP_K


@dataclass(frozen=True)
class OvernightTrainingResult:
    model: OvernightModel
    label_rows: int  # Selected Rank cohort with verified labels.
    scored_rows: int
    warmup_rows: int
    provider_labeled_rows: int  # All fetched ticker-sessions, before Rank/date filters.
    provider_candidate_rows: int  # Provider-labeled plus provider-excluded rows.
    provider_excluded_by_reason: dict[str, int]


def fetch_start_with_prior_sessions(requested_start: date, *, calendar=None) -> date:
    """Include six verified XNYS closes before the first requested label day."""
    exchange = calendar if calendar is not None else xcals.get_calendar("XNYS")
    if getattr(exchange, "name", None) != "XNYS":
        raise ValueError("overnight history requires the XNYS calendar")
    first = exchange.date_to_session(requested_start.isoformat(), direction="next")
    prior = first
    for _ in range(PRIOR_CLOSE_COUNT):
        prior = exchange.previous_session(prior)
    return pd.Timestamp(prior).date()


def build_verified_overnight_rows(
    verified_by_ticker: dict[str, VerifiedOvernightBars], contract: PriceContract,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pooled close→next-open labels only from source/basis/action-checked bars."""
    frames = []
    exclusions: dict[str, int] = {}
    for ticker, verified in sorted(verified_by_ticker.items()):
        built = build_overnight_training_frame(verified.history, verified.provenance, contract)
        for reason, count in built.exclusion_counts.items():
            exclusions[reason] = exclusions.get(reason, 0) + count
        if built.frame.empty:
            continue
        frame = built.frame.reset_index(names="date")
        frame.insert(0, "ticker", ticker)
        frames.append(frame)
    if not frames:
        raise ValueError(f"no verified overnight training rows: {exclusions}")
    return pd.concat(frames, ignore_index=True), exclusions


def train_and_persist_overnight(
    verified_by_ticker: dict[str, VerifiedOvernightBars],
    day_training_frame: pd.DataFrame,
    *,
    fit_specs: list[ModelSpec],
    rank_spec: ModelSpec,
    contract: PriceContract,
    model_store: ModelStore,
    tracking_dir: Path,
    n_splits: int = 4,
    day_model_source: dict[str, object] | None = None,
    serving_fit_model: Ensemble | None = None,
    serving_rank_model: Ensemble | None = None,
    day_model_trained_through: date | None = None,
    historical_day_scores: pd.DataFrame | None = None,
    rank_top_k: int = RANK_TOP_K,
    label_start: date | None = None,
    label_end: date | None = None,
    ticker_whitelist: set[str] | None = None,
) -> OvernightTrainingResult:
    """Train only on reconstructed as-of Rank picks with verified labels."""
    label_rows, exclusions = build_verified_overnight_rows(verified_by_ticker, contract)
    provider_labeled_rows = len(label_rows)
    provider_candidate_rows = provider_labeled_rows + sum(exclusions.values())
    if label_start is not None and label_end is not None and label_start > label_end:
        raise ValueError("requested overnight label range is reversed")
    label_dates = pd.to_datetime(label_rows["date"])
    if label_start is not None:
        label_rows = label_rows.loc[label_dates >= pd.Timestamp(label_start)]
    if label_end is not None:
        label_rows = label_rows.loc[pd.to_datetime(label_rows["date"]) <= pd.Timestamp(label_end)]
    day_scores = historical_day_scores if historical_day_scores is not None else generate_historical_day_scores(
        day_training_frame,
        fit_specs=fit_specs,
        rank_spec=rank_spec,
        tracking_dir=tracking_dir,
        n_splits=n_splits,
    )
    cohort = select_ranked_cohort(day_scores, day_training_frame, top_k=rank_top_k)
    scored_dates = set(cohort["date"])
    first_scored_date = cohort["date"].min()
    unscored_dates = ~pd.to_datetime(label_rows["date"]).isin(scored_dates)
    if (pd.to_datetime(label_rows.loc[unscored_dates, "date"]) >= first_scored_date).any():
        raise ValueError("verified overnight rows are missing non-warmup same-day scores")
    warmup_rows = int(unscored_dates.sum())
    eligible = ranked_labeled_rows(label_rows, cohort)
    if ticker_whitelist is not None:
        eligible = eligible.loc[eligible["ticker"].isin(ticker_whitelist)]
    if eligible.empty:
        raise ValueError("no verified overnight labels overlap as-of Rank picks")
    manifest = make_ranked_cohort_manifest(
        cohort, label_rows, label_start=label_start, label_end=label_end,
        ticker_whitelist=ticker_whitelist,
    )
    manifest_path, manifest_sha256 = save_ranked_cohort_manifest(manifest, tracking_dir)
    scored = attach_historical_day_scores(eligible, day_scores)
    model = train_overnight_model(scored, contract, n_splits=n_splits)
    model.day_model_source = {
        **(day_model_source or {}), "cohort_source": COHORT_SOURCE, "rank_top_k": rank_top_k,
        "cohort_universe_limit": COHORT_UNIVERSE_LIMIT,
        "ticker_whitelist": sorted(ticker_whitelist) if ticker_whitelist is not None else None,
        "cohort_manifest_relative_path": manifest_path.relative_to(tracking_dir).as_posix(),
        "cohort_manifest_sha256": manifest_sha256,
    }
    if any(value is not None for value in (serving_fit_model, serving_rank_model, day_model_trained_through)):
        if serving_fit_model is None or serving_rank_model is None or day_model_trained_through is None:
            raise ValueError("serving morning models and their training cutoff must be supplied together")
        model.day_fit_model = serving_fit_model
        model.day_rank_model = serving_rank_model
        model.day_model_trained_through = day_model_trained_through
    model_store.write(MODEL_NAME, model)
    return OvernightTrainingResult(
        model=model,
        label_rows=len(eligible),
        scored_rows=len(scored),
        warmup_rows=warmup_rows,
        provider_labeled_rows=provider_labeled_rows,
        provider_candidate_rows=provider_candidate_rows,
        provider_excluded_by_reason=exclusions,
    )


def _specs_from_saved_models(fit: Ensemble, rank: Ensemble) -> tuple[list[ModelSpec], ModelSpec]:
    """Retrain the chosen same-day architecture in historical folds."""
    if not fit.members or len(fit.members) != len(fit.weights) or not rank.members or len(rank.members) != 1:
        raise ValueError("saved Fit and one-member Rank artifacts are required")
    required_svm = {"svr_oof_pred", "svc_direction_margin"}
    if required_svm - set(getattr(fit, "stacked_svm_estimators", None) or {}):
        raise ValueError("selected Fit artifact lacks the requested SVR and direction-SVC outputs")
    fit_specs = [
        ModelSpec(member.model_type, included_features=set(member.feature_names), weight=weight)
        for member, weight in zip(fit.members, fit.weights)
    ]
    rank_spec = ModelSpec(rank.members[0].model_type, included_features=set(rank.members[0].feature_names))
    if rank_spec.model_type != "lightgbm_rank":
        raise ValueError("saved Rank artifact is not the same-day rank model")
    return fit_specs, rank_spec


def main(argv: list[str] | None = None) -> None:
    """Explicit, bounded research run; no automatic market-day retraining."""
    parser = argparse.ArgumentParser(description="Train close-conditioned next-open LightGBM")
    parser.add_argument(
        "--tickers", nargs="+", help="Optional whitelist applied after full-universe Rank top-20 selection",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--day-universe", choices=("all",), default="all",
        help="Historical Fit/Rank folds must use the full universe",
    )
    parser.add_argument("--output-model-dir", type=Path, help="Write the model in this directory instead of the default model store")
    parser.add_argument("--tracking-dir", type=Path, help="Separate MLflow directory for this experiment")
    args = parser.parse_args(argv)
    if args.tickers and (len(args.tickers) != len(set(args.tickers)) or len(args.tickers) > 200):
        parser.error("provide 1–200 unique ticker whitelist entries")
    if args.start >= args.end or args.folds < 2:
        parser.error("start must precede end and folds must be at least two")

    source_model_store = ModelStore()
    selected_run_id = TrainingConfigStore().read().selected_run_id
    fit_name = f"{DAY_MODEL_NAME}_{selected_run_id}" if selected_run_id else DAY_MODEL_NAME
    from stock_picker.training.rank_model import RANK_MODEL_NAME

    if not source_model_store.exists(fit_name) or not source_model_store.exists(RANK_MODEL_NAME):
        raise ValueError("selected same-day Fit or Rank model is missing")
    fit, rank = source_model_store.read(fit_name), source_model_store.read(RANK_MODEL_NAME)
    fit_specs, rank_spec = _specs_from_saved_models(fit, rank)
    day_tickers = UniverseStore().active_tickers()
    day_training = _load_pooled_dataset(day_tickers, PriceStore(), FeatureStore())
    needed = set().union(*(set(member.feature_names) for member in (*fit.members, *rank.members)))
    for estimator in (getattr(fit, "stacked_svm_estimators", None) or {}).values():
        needed.update(estimator.feature_names)
    missing = needed - set(day_training) - set(STACKED_SVM_COLUMNS)
    if missing:
        raise ValueError(f"historical same-day inputs are missing: {sorted(missing)}")

    client = MassiveOvernightClient()
    # Choose Rank names before the expensive, raw/action-checked provider calls.
    # The last requested session has no next-open label within this fetch range.
    historical_scores = generate_historical_day_scores(
        day_training, fit_specs=fit_specs, rank_spec=rank_spec,
        tracking_dir=args.tracking_dir or data_root() / "mlruns", n_splits=args.folds,
    )
    cohort = select_ranked_cohort(historical_scores, day_training)
    eligible_dates = (cohort["date"] >= pd.Timestamp(args.start)) & (cohort["date"] < pd.Timestamp(args.end))
    needed_tickers = set(cohort.loc[eligible_dates, "ticker"])
    if args.tickers:
        needed_tickers &= set(args.tickers)
    if not needed_tickers:
        raise ValueError("no dated Rank picks are eligible for a next-open label in this range")
    fetch_start = fetch_start_with_prior_sessions(args.start)
    verified = {ticker: client.fetch(ticker, fetch_start, args.end) for ticker in sorted(needed_tickers)}
    output_store = ModelStore(data_dir=args.output_model_dir) if args.output_model_dir else source_model_store
    result = train_and_persist_overnight(
        verified,
        day_training,
        fit_specs=fit_specs,
        rank_spec=rank_spec,
        contract=PriceContract("massive", "raw", "massive_actions"),
        model_store=output_store,
        tracking_dir=args.tracking_dir or data_root() / "mlruns",
        n_splits=args.folds,
        day_model_source={
            "fit_artifact": fit_name,
            "rank_artifact": RANK_MODEL_NAME,
            "day_universe_mode": args.day_universe,
            "day_universe_tickers": len(day_tickers),
        },
        serving_fit_model=fit,
        serving_rank_model=rank,
        # This is conservative if an existing saved estimator was fitted
        # earlier than the frame we loaded; it can only delay serving.
        day_model_trained_through=pd.to_datetime(day_training["date"]).max().date(),
        historical_day_scores=historical_scores,
        label_start=args.start,
        label_end=args.end,
        ticker_whitelist=set(args.tickers) if args.tickers else None,
    )
    print(json.dumps({
        "artifact": MODEL_NAME,
        "features": list(result.model.feature_columns),
        "label_rows": result.label_rows,
        "scored_rows": result.scored_rows,
        "warmup_rows": result.warmup_rows,
        "provider_labeled_rows": result.provider_labeled_rows,
        "provider_candidate_rows": result.provider_candidate_rows,
        "provider_excluded_by_reason": result.provider_excluded_by_reason,
        "folds": [vars(fold) for fold in result.model.folds],
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
