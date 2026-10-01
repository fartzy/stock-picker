"""Entrypoint: train the day-session return model, holding out a few tickers entirely
to test whether the model's signal generalizes to stocks it has never seen -- not just
future dates for tickers it has already seen (that's what walk-forward validates).
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_picker.features.pruning import pruned_features
from stock_picker.features.selection import selected_features
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.log import get_logger
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.training_run_store import TrainingRunRecord, TrainingRunStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import sweep_thresholds
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.ensemble import (
    ModelSpec,
    ensemble_feature_names,
    evaluate_ensemble,
    partition_model_specs,
    predict_ensemble,
    selected_model_specs,
)
from stock_picker.training.model import NON_FEATURE_COLUMNS, EvaluationMetrics, train_logistic_regression
from stock_picker.training.splits import select_holdout_tickers
from stock_picker.training.train import run_walk_forward

MODEL_NAME = "day_session_return"
# Persisted separately from MODEL_NAME's ensemble -- see the diagnostic-fit
# note in run_training() below for why logistic regression isn't a member of
# the Ensemble itself.
DIAGNOSTIC_MODEL_NAME = f"{MODEL_NAME}_logistic_diagnostic"
# The default ensemble composition when no UI selection has been made (see
# ensemble.py's selected_model_specs()) -- a developer can still edit this
# directly to change the out-of-the-box defaults. RandomForest was tried
# alongside LightGBM here (see training/tune_experiment.py) and dropped: a
# weight search over solo-vs-blended configurations picked solo LightGBM
# outright across every blend tried, so it never earned a place -- the UI's
# composable model picker can still add it back in for experimentation.
DEFAULT_MODEL_SPECS = [ModelSpec("lightgbm")]

logger = get_logger(__name__)


@dataclass
class TrainingSummary:
    """What run_training() below actually produces -- shared by the CLI
    entrypoint and, via training/job.py, the /api/training/run endpoint.

    train_tickers/holdout_tickers/date_range/resolved_features/model_specs
    are this run's provenance -- what storage/training_run_store.py persists
    so a past run can be inspected later, not just today's job status.
    date_range covers train_dataset (what was actually fit on), not the
    holdout set, which is eval data rather than "fed to" training.
    """

    fold_metrics: list[EvaluationMetrics]
    holdout_metrics: EvaluationMetrics | None
    threshold_sweep: list[dict] | None
    train_tickers: list[str]
    holdout_tickers: list[str]
    date_range: tuple[str, str]
    resolved_features: list[str]
    model_specs: list[dict]


def _date_range(frame: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(frame["date"])
    return (str(dates.min().date()), str(dates.max().date()))


def _load_pooled_dataset(
    tickers: list[str], price_store: PriceStore, feature_store: FeatureStore
) -> pd.DataFrame:
    histories = {}
    features_by_ticker = {}
    for ticker in tickers:
        try:
            histories[ticker] = price_store.read(ticker)
            features_by_ticker[ticker] = feature_store.read(ticker)
        except FileNotFoundError:
            # Active in UniverseStore doesn't guarantee price/feature data exists
            # for it (e.g. a transient ingestion failure) -- skip rather than
            # crash the whole run over one ticker.
            logger.warning("skipping %s: missing price or feature data", ticker)
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def run_training(
    included_features: set[str] | None = None,
    model_specs: list[ModelSpec] | None = None,
    run_id: str | None = None,
    stack_direction_margin: bool = True,
) -> TrainingSummary:
    """Runs one full walk-forward + holdout + threshold-sweep pass and persists
    the final ensemble, returning a plain-JSON-serializable summary. Shared by
    the CLI entrypoint (`main()` below) and `api/routes.py`'s `/api/training/run`
    endpoint, so there's exactly one training path regardless of who triggers it.

    `included_features`, if given, restricts every return-ensemble member
    (and the standalone diagnostic fit below) to that exact set (still
    always minus the pruned set -- see `feature_columns()`'s precedence).
    The direction SVC is deliberately different: when the LightGBM margin
    is active it uses every unpruned raw feature, even if LightGBM has a
    narrow positive selection. `None` means every feature, subject to
    pruning only. At least one unpruned raw feature is required when using a
    positive selection, because the standalone diagnostic and Rank models
    cannot train on a model-derived margin alone.

    `stack_direction_margin` defaults on for production Fit. Research scripts
    that selected a winner using raw-feature folds pass False when
    materializing that winner so the archived model matches its comparison.

    `model_specs`, if given, is the composable composition chosen via the UI
    (see `ensemble.py`'s `selected_model_specs()`); `None` falls back to
    `DEFAULT_MODEL_SPECS` and also trains the ListFold rank model. `lightgbm_rank` is
    peeled out of the return blender (see `partition_model_specs`) and
    written to `day_session_return_rank.pkl`. Each spec's own
    `excluded_features`/`included_features` are overwritten here with the
    current pruned/selected sets, since specs coming from persisted UI
    config only carry `model_type`/`weight` -- feature selection is applied
    uniformly to every model in the ensemble (per-model feature subsets are
    deferred).

    `run_id`, if given, also archives this run's ensemble under its own name
    (in addition to the plain overwrite below, which stays "whatever's
    latest") -- the caller already generates this same id for its
    `TrainingRunRecord`, so the archived model and its Run History entry
    share one identifier. See `storage/training_config_store.py`'s
    `selected_run_id` and `training/buy_signal.py` for how a past run gets
    picked back up for live inference.
    """
    tickers = UniverseStore().active_tickers()
    holdout_ticker_set = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout_ticker_set]
    holdout_tickers = [t for t in tickers if t in holdout_ticker_set]

    excluded_features = pruned_features()
    if included_features is not None and not (
        included_features - excluded_features - set(STACKED_SVM_COLUMNS) - NON_FEATURE_COLUMNS
    ):
        raise ValueError(
            "positive feature selection must include at least one unpruned raw feature; "
            "the direction SVC margin alone cannot train diagnostic or Rank models"
        )
    price_store = PriceStore()
    feature_store = FeatureStore()
    predictive_specs, wants_rank = partition_model_specs(model_specs)
    base_specs = predictive_specs if predictive_specs is not None else DEFAULT_MODEL_SPECS
    specs = [
        ModelSpec(
            spec.model_type,
            params=spec.params,
            weight=spec.weight,
            excluded_features=excluded_features,
            included_features=included_features,
        )
        for spec in base_specs
    ]
    resolved_specs = [{"model_type": s.model_type, "weight": s.weight, "params": s.params} for s in specs]
    if wants_rank:
        resolved_specs.append({"model_type": "lightgbm_rank", "weight": 1.0, "params": None})
    # Surfaced here, before the walk-forward/fit calls that can raise, so a
    # failed run still leaves this provenance in the server log even though
    # storage/training_run_store.py can't record it for a run that never
    # reaches a TrainingSummary (see training/job.py).
    logger.info(
        "training on %s tickers, holding out %s: %s",
        len(train_tickers),
        len(holdout_tickers),
        resolved_specs,
    )

    train_dataset = _load_pooled_dataset(train_tickers, price_store, feature_store)
    if included_features is not None:
        # The catalog describes intended pipeline columns, but a few are not
        # yet materialized in persisted training data. A named raw feature
        # only counts if this run can actually train on it.
        available_raw = set(train_dataset.columns) - NON_FEATURE_COLUMNS - set(STACKED_SVM_COLUMNS)
        if not ((included_features - excluded_features) & available_raw):
            raise ValueError(
                "positive feature selection must include at least one unpruned raw feature "
                "present in the training dataset"
            )

    # Production Fit enables the derived feature; raw-feature research
    # winners explicitly opt out so materialization matches their search.
    fold_results = run_walk_forward(
        train_dataset, specs=specs, stack_direction_margin=stack_direction_margin
    )
    for result in fold_results:
        logger.info("fold %s: %s", result.fold, result.metrics)

    final_ensemble = fold_results[-1].model

    # Fit alone sees the model-derived margin, while all other members see
    # raw features only. Include the saved SVC's raw inputs too, so the run
    # record describes everything the archived predictor requires to score.
    resolved_features = sorted(ensemble_feature_names(final_ensemble))

    # Standalone diagnostic fit, not an Ensemble member: logistic regression
    # predicts binary direction, a unit incompatible with the continuous
    # return the ensemble blends, so it can't be weighted-averaged in. Fit on
    # the full pooled training set (not just the last fold) since it's purely
    # for its own coefficient-based importance view, not for prediction.
    diagnostic_model = train_logistic_regression(
        train_dataset, excluded_features=excluded_features, included_features=included_features
    )

    fold_metrics = [result.metrics for result in fold_results]
    holdout_metrics = None
    threshold_sweep = None
    if holdout_tickers:
        holdout_dataset = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
        holdout_metrics = evaluate_ensemble(final_ensemble, holdout_dataset)
        logger.info("holdout tickers %s: %s", holdout_tickers, holdout_metrics)

        predicted = pd.Series(predict_ensemble(final_ensemble, holdout_dataset), index=holdout_dataset.index)
        actual = holdout_dataset[LABEL_COLUMN]
        sweep = sweep_thresholds(predicted, actual, n_days=holdout_dataset["date"].nunique())
        logger.info("%s", sweep.to_string(index=False))

        # DataFrame.to_dict() leaves numpy scalar types in place (not JSON-
        # serializable as-is) -- round-tripping through to_json()/json.loads()
        # is pandas' own well-tested path for native Python types instead.
        threshold_sweep = json.loads(sweep.to_json(orient="records"))

    if wants_rank:
        from stock_picker.training.rank_model import train_and_persist_rank_model

        logger.info("training ListFold rank model (parallel pickle, not blended into the return ensemble)")
        train_and_persist_rank_model(included_features=included_features)

    summary = TrainingSummary(
        fold_metrics=fold_metrics,
        holdout_metrics=holdout_metrics,
        threshold_sweep=threshold_sweep,
        train_tickers=train_tickers,
        holdout_tickers=holdout_tickers,
        date_range=_date_range(train_dataset),
        resolved_features=resolved_features,
        model_specs=resolved_specs,
    )
    # Publish only after every evaluation and companion model has succeeded.
    # A failed holdout/Rank/diagnostic fit must not replace a working Fit
    # model; write the run-specific archive before changing `latest`.
    model_store = ModelStore()
    model_store.write(DIAGNOSTIC_MODEL_NAME, diagnostic_model)
    if run_id:
        model_store.write(f"{MODEL_NAME}_{run_id}", final_ensemble)
    model_store.write(MODEL_NAME, final_ensemble)
    return summary


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _duration_seconds(started_at: str, completed_at: str) -> float:
    return (datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds()


def _git_commit() -> str | None:
    # cwd mirrors storage/paths.py's data_root() reasoning: `bazel run` sandboxes
    # the process's actual cwd to a runfiles dir with no `.git` in it, but sets
    # BUILD_WORKING_DIRECTORY to the directory the user invoked bazel from.
    cwd = os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return None


def main() -> None:
    # Mirrors training/job.py's TrainingJob._run() recording -- the API's
    # "Run training" button goes through job.py, but this CLI entrypoint
    # (bazel run //python/stock_picker/training:main, also what ONBOARDING.md
    # tells a fresh clone to run) previously trained and persisted a model
    # without ever appending a TrainingRunRecord, so a run triggered this way
    # silently never showed up in Run History even though it really did
    # retrain and overwrite the live model.
    run_store = TrainingRunStore()
    started_at = _now()
    run_id = uuid.uuid4().hex
    try:
        result = run_training(
            included_features=selected_features(), model_specs=selected_model_specs(), run_id=run_id
        )
    except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised so the CLI still exits non-zero
        completed_at = _now()
        run_store.append(
            TrainingRunRecord(
                run_id=run_id,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=_duration_seconds(started_at, completed_at),
                git_commit=_git_commit(),
                error=str(exc),
            )
        )
        raise
    completed_at = _now()
    run_store.append(
        TrainingRunRecord(
            run_id=run_id,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=_duration_seconds(started_at, completed_at),
            git_commit=_git_commit(),
            train_tickers=result.train_tickers,
            holdout_tickers=result.holdout_tickers,
            date_range=result.date_range,
            resolved_features=result.resolved_features,
            model_specs=result.model_specs,
            fold_metrics=[asdict(m) for m in result.fold_metrics],
            holdout_metrics=asdict(result.holdout_metrics) if result.holdout_metrics is not None else None,
            threshold_sweep=result.threshold_sweep,
        )
    )


if __name__ == "__main__":
    main()
