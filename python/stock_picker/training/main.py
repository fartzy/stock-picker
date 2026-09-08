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
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.training_run_store import TrainingRunRecord, TrainingRunStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import sweep_thresholds
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.ensemble import ModelSpec, evaluate_ensemble, predict_ensemble, selected_model_specs
from stock_picker.training.model import EvaluationMetrics, train_logistic_regression
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
            print(f"skipping {ticker}: missing price or feature data")
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def run_training(
    included_features: set[str] | None = None,
    model_specs: list[ModelSpec] | None = None,
    run_id: str | None = None,
) -> TrainingSummary:
    """Runs one full walk-forward + holdout + threshold-sweep pass and persists
    the final ensemble, returning a plain-JSON-serializable summary. Shared by
    the CLI entrypoint (`main()` below) and `api/routes.py`'s `/api/training/run`
    endpoint, so there's exactly one training path regardless of who triggers it.

    `included_features`, if given, restricts every ensemble member (and the
    standalone diagnostic fit below) to that exact set (still always minus
    the pruned set -- see `feature_columns()`'s precedence). `None` means
    "every feature, subject to pruning only," same as before this parameter
    existed.

    `model_specs`, if given, is the composable ensemble composition chosen via
    the UI (see `ensemble.py`'s `selected_model_specs()`); `None` falls back
    to `DEFAULT_MODEL_SPECS`. Each spec's own `excluded_features`/
    `included_features` are overwritten here with the current pruned/selected
    sets, since specs coming from persisted UI config only carry
    `model_type`/`weight` -- feature selection is applied uniformly to every
    model in the ensemble (per-model feature subsets are deferred).

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

    price_store = PriceStore()
    feature_store = FeatureStore()
    excluded_features = pruned_features()
    base_specs = model_specs if model_specs is not None else DEFAULT_MODEL_SPECS
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
    # Surfaced here, before the walk-forward/fit calls that can raise, so a
    # failed run still leaves this provenance in the server log even though
    # storage/training_run_store.py can't record it for a run that never
    # reaches a TrainingSummary (see training/job.py).
    print(f"training on {len(train_tickers)} tickers, holding out {len(holdout_tickers)}: {resolved_specs}")

    train_dataset = _load_pooled_dataset(train_tickers, price_store, feature_store)

    fold_results = run_walk_forward(train_dataset, specs=specs)
    for result in fold_results:
        print(f"fold {result.fold}: {result.metrics}")

    final_ensemble = fold_results[-1].model
    ModelStore().write(MODEL_NAME, final_ensemble)
    if run_id:
        ModelStore().write(f"{MODEL_NAME}_{run_id}", final_ensemble)

    # Every member trains against the same train_dataset with the same
    # excluded_features/included_features (applied uniformly above), so
    # today they always resolve to the same feature_names -- an emergent
    # property of that uniform filtering, not an enforced invariant. This
    # assertion is the tripwire: per-model feature subsets are flagged as
    # deferred future work in this function's own docstring above, and
    # would break it.
    resolved_features = final_ensemble.members[0].feature_names
    assert all(member.feature_names == resolved_features for member in final_ensemble.members)

    # Standalone diagnostic fit, not an Ensemble member: logistic regression
    # predicts binary direction, a unit incompatible with the continuous
    # return the ensemble blends, so it can't be weighted-averaged in. Fit on
    # the full pooled training set (not just the last fold) since it's purely
    # for its own coefficient-based importance view, not for prediction.
    diagnostic_model = train_logistic_regression(
        train_dataset, excluded_features=excluded_features, included_features=included_features
    )
    ModelStore().write(DIAGNOSTIC_MODEL_NAME, diagnostic_model)

    fold_metrics = [result.metrics for result in fold_results]

    if not holdout_tickers:
        return TrainingSummary(
            fold_metrics=fold_metrics,
            holdout_metrics=None,
            threshold_sweep=None,
            train_tickers=train_tickers,
            holdout_tickers=holdout_tickers,
            date_range=_date_range(train_dataset),
            resolved_features=resolved_features,
            model_specs=resolved_specs,
        )

    holdout_dataset = _load_pooled_dataset(holdout_tickers, price_store, feature_store)
    holdout_metrics = evaluate_ensemble(final_ensemble, holdout_dataset)
    print(f"holdout tickers {holdout_tickers}: {holdout_metrics}")

    predicted = pd.Series(predict_ensemble(final_ensemble, holdout_dataset), index=holdout_dataset.index)
    actual = holdout_dataset[LABEL_COLUMN]
    sweep = sweep_thresholds(predicted, actual, n_days=holdout_dataset["date"].nunique())
    print(sweep.to_string(index=False))

    # DataFrame.to_dict() leaves numpy scalar types in place (not JSON-
    # serializable as-is) -- round-tripping through to_json()/json.loads()
    # is pandas' own well-tested path for native Python types instead.
    threshold_sweep = json.loads(sweep.to_json(orient="records"))
    return TrainingSummary(
        fold_metrics=fold_metrics,
        holdout_metrics=holdout_metrics,
        threshold_sweep=threshold_sweep,
        train_tickers=train_tickers,
        holdout_tickers=holdout_tickers,
        date_range=_date_range(train_dataset),
        resolved_features=resolved_features,
        model_specs=resolved_specs,
    )


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
