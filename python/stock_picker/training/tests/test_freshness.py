from datetime import date

import pandas as pd

from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.training_config_store import TrainingConfigStore
from stock_picker.storage.training_run_store import TrainingRunRecord, TrainingRunStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.freshness import pipeline_freshness


def _completed_run(run_id: str, through: str) -> TrainingRunRecord:
    return TrainingRunRecord(
        run_id=run_id,
        status="completed",
        started_at=f"{through}T17:00:00-05:00",
        completed_at=f"{through}T17:20:00-05:00",
        duration_seconds=1200.0,
        date_range=("2025-09-10", through),
    )


def test_ready_when_features_and_model_reach_the_last_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.freshness.last_completed_session_date",
        lambda: date(2026, 9, 10),
    )
    features = FeatureStore(data_dir=tmp_path / "features")
    universe = UniverseStore(data_dir=tmp_path / "universe")
    runs = TrainingRunStore(data_dir=tmp_path / "runs")
    config = TrainingConfigStore(data_dir=tmp_path / "config")
    universe.sync({"AAPL": "manual"})
    features.write(
        "AAPL",
        pd.DataFrame({"overnight_gap": [0.01]}, index=pd.DatetimeIndex(["2026-09-10"])),
    )
    runs.append(_completed_run("abc", "2026-09-10"))

    result = pipeline_freshness(
        as_of=date(2026, 9, 11),
        feature_store=features,
        universe_store=universe,
        run_store=runs,
        config_store=config,
    )

    assert result.ready_for_inference
    assert result.features_ok
    assert result.model_ok
    assert result.live_run_id == "abc"


def test_stale_model_trained_through_monday_cannot_score_thursday(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stock_picker.training.freshness.last_completed_session_date",
        lambda: date(2026, 9, 10),
    )
    features = FeatureStore(data_dir=tmp_path / "features")
    universe = UniverseStore(data_dir=tmp_path / "universe")
    runs = TrainingRunStore(data_dir=tmp_path / "runs")
    config = TrainingConfigStore(data_dir=tmp_path / "config")
    universe.sync({"AAPL": "manual"})
    features.write(
        "AAPL",
        pd.DataFrame({"overnight_gap": [0.01]}, index=pd.DatetimeIndex(["2026-09-10"])),
    )
    runs.append(_completed_run("old", "2026-09-08"))

    result = pipeline_freshness(
        as_of=date(2026, 9, 11),
        feature_store=features,
        universe_store=universe,
        run_store=runs,
        config_store=config,
    )

    assert result.features_ok
    assert not result.model_ok
    assert not result.ready_for_inference
    assert "full retrain" in result.detail
