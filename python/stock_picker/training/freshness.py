"""Whether last night's pipeline is current enough for this morning's score.

LightGBM is not incrementally patched with two new days -- nightly full
retrain *is* the update. Inference still needs (1) a feature snapshot from
the last completed session and (2) a model whose training date_range
reaches that same session. A model trained through Monday cannot be used
on Thursday morning (TTL is 2 weekdays -- same as inference.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_picker.features.registry import DEFAULT_TTL_DAYS, check_freshness
from stock_picker.ingestion.session import last_completed_session_date
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.storage.training_run_store import TrainingRunRecord, TrainingRunStore
from stock_picker.storage.universe_store import UniverseStore


@dataclass
class PipelineFreshness:
    as_of: str
    last_completed_session: str
    feature_snapshot_date: str | None
    features_ok: bool
    feature_age_weekdays: int | None
    model_trained_through: str | None
    model_ok: bool
    model_age_weekdays: int | None
    ready_for_inference: bool
    live_run_id: str | None
    detail: str


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _live_run(run_store: TrainingRunStore, config_store: TrainingConfigStore) -> TrainingRunRecord | None:
    selected = config_store.read().selected_run_id
    completed = [record for record in run_store.read_all() if record.status == "completed"]
    if selected:
        for record in completed:
            if record.run_id == selected:
                return record
        return None
    return completed[0] if completed else None


def _feature_snapshot_date(feature_store: FeatureStore, universe_store: UniverseStore) -> date | None:
    tickers = universe_store.active_tickers()
    sample = "AAPL" if "AAPL" in tickers else (tickers[0] if tickers else None)
    if sample is None:
        return None
    try:
        features = feature_store.read(sample)
    except FileNotFoundError:
        return None
    if features.empty:
        return None
    return features.index[-1].date()


def pipeline_freshness(
    as_of: date | None = None,
    feature_store: FeatureStore | None = None,
    universe_store: UniverseStore | None = None,
    run_store: TrainingRunStore | None = None,
    config_store: TrainingConfigStore | None = None,
) -> PipelineFreshness:
    as_of = as_of or date.today()
    last_session = last_completed_session_date()
    feature_store = feature_store or FeatureStore()
    universe_store = universe_store or UniverseStore()
    run_store = run_store or TrainingRunStore()
    config_store = config_store or TrainingConfigStore()

    snapshot = _feature_snapshot_date(feature_store, universe_store)
    if snapshot is None:
        feature_check = None
        features_ok = False
        feature_age = None
    else:
        feature_check = check_freshness(DEFAULT_TTL_DAYS, snapshot, as_of)
        features_ok = feature_check.ok
        feature_age = feature_check.age_days

    live = _live_run(run_store, config_store)
    trained_through = None
    if live and live.date_range:
        trained_through = _parse_iso_date(live.date_range[1])
    if trained_through is None:
        model_ok = False
        model_age = None
    else:
        model_check = check_freshness(DEFAULT_TTL_DAYS, trained_through, as_of)
        model_ok = model_check.ok
        model_age = model_check.age_days

    ready = features_ok and model_ok
    if snapshot is None:
        detail = "No feature snapshot on disk -- run the nightly job (prices + features + retrain)."
    elif not features_ok:
        detail = (
            f"Features stop at {snapshot.isoformat()}, last completed session is "
            f"{last_session.isoformat()} -- rebuild features before scoring."
        )
    elif trained_through is None:
        detail = "No completed training run -- train a model before scoring."
    elif not model_ok:
        detail = (
            f"Live model was trained through {trained_through.isoformat()}, last completed "
            f"session is {last_session.isoformat()} -- full retrain (not a patch) before scoring."
        )
    else:
        detail = (
            f"Features through {snapshot.isoformat()}, model trained through "
            f"{trained_through.isoformat()} -- ready for this morning's opens."
        )

    return PipelineFreshness(
        as_of=as_of.isoformat(),
        last_completed_session=last_session.isoformat(),
        feature_snapshot_date=snapshot.isoformat() if snapshot else None,
        features_ok=features_ok,
        feature_age_weekdays=feature_age,
        model_trained_through=trained_through.isoformat() if trained_through else None,
        model_ok=model_ok,
        model_age_weekdays=model_age,
        ready_for_inference=ready,
        live_run_id=live.run_id if live else None,
        detail=detail,
    )
