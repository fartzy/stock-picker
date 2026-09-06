import threading
import time

from stock_picker.storage.training_run_store import TrainingRunStore
from stock_picker.training.job import TrainingJob
from stock_picker.training.main import TrainingSummary
from stock_picker.training.model import EvaluationMetrics


def _wait_until(predicate, timeout=2.0):
    """Polls until predicate() is true or the timeout elapses -- avoids a
    flaky fixed sleep while waiting for the background thread to progress."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "timed out waiting for condition"


def _fake_summary():
    return TrainingSummary(
        fold_metrics=[EvaluationMetrics(mae=0.01, directional_accuracy=0.5, n_test_rows=100)],
        holdout_metrics=EvaluationMetrics(mae=0.02, directional_accuracy=0.55, n_test_rows=200),
        threshold_sweep=[{"threshold": 0.005, "n_trades": 10, "hit_rate": 0.6}],
        train_tickers=["AAPL", "MSFT"],
        holdout_tickers=["GOOG"],
        date_range=("2026-01-01", "2026-01-31"),
        resolved_features=["return_1d"],
        model_specs=[{"model_type": "lightgbm", "weight": 1.0, "params": None}],
    )


def test_status_starts_idle(tmp_path):
    job = TrainingJob(
        train_fn=lambda included_features, model_specs: None, run_store=TrainingRunStore(data_dir=tmp_path)
    )

    assert job.status().status == "idle"


def test_start_runs_train_fn_and_reports_completed(tmp_path):
    summary = _fake_summary()
    job = TrainingJob(
        train_fn=lambda included_features, model_specs: summary,
        run_store=TrainingRunStore(data_dir=tmp_path),
    )

    started = job.start()

    assert started is True
    _wait_until(lambda: job.status().status == "completed")
    status = job.status()
    assert status.result == summary
    assert status.error is None
    assert status.started_at is not None
    assert status.completed_at is not None


def test_start_passes_included_features_through_to_train_fn(tmp_path):
    received = {}

    def fake_train(included_features, model_specs):
        received["included_features"] = included_features
        return _fake_summary()

    job = TrainingJob(train_fn=fake_train, run_store=TrainingRunStore(data_dir=tmp_path))
    job.start(included_features={"return_1d"})

    _wait_until(lambda: job.status().status == "completed")
    assert received["included_features"] == {"return_1d"}


def test_start_passes_model_specs_through_to_train_fn(tmp_path):
    received = {}

    def fake_train(included_features, model_specs):
        received["model_specs"] = model_specs
        return _fake_summary()

    job = TrainingJob(train_fn=fake_train, run_store=TrainingRunStore(data_dir=tmp_path))
    job.start(model_specs=["fake spec"])

    _wait_until(lambda: job.status().status == "completed")
    assert received["model_specs"] == ["fake spec"]


def test_start_returns_false_while_a_run_is_already_in_progress(tmp_path):
    release = threading.Event()

    def fake_train(included_features, model_specs):
        release.wait(timeout=2.0)
        return _fake_summary()

    job = TrainingJob(train_fn=fake_train, run_store=TrainingRunStore(data_dir=tmp_path))

    first_started = job.start()
    _wait_until(lambda: job.status().status == "running")
    second_started = job.start()

    assert first_started is True
    assert second_started is False

    release.set()
    _wait_until(lambda: job.status().status == "completed")


def test_a_train_fn_exception_reports_failed_with_the_error_message(tmp_path):
    def failing_train(included_features, model_specs):
        raise ValueError("boom")

    job = TrainingJob(train_fn=failing_train, run_store=TrainingRunStore(data_dir=tmp_path))
    job.start()

    _wait_until(lambda: job.status().status == "failed")
    status = job.status()
    assert status.error == "boom"
    assert status.result is None


def test_a_completed_run_appends_a_training_run_record_with_full_provenance(tmp_path):
    run_store = TrainingRunStore(data_dir=tmp_path)
    job = TrainingJob(train_fn=lambda included_features, model_specs: _fake_summary(), run_store=run_store)

    job.start()

    _wait_until(lambda: job.status().status == "completed")
    [record] = run_store.read_all()
    assert record.status == "completed"
    assert record.train_tickers == ["AAPL", "MSFT"]
    assert record.holdout_tickers == ["GOOG"]
    assert record.date_range == ("2026-01-01", "2026-01-31")
    assert record.resolved_features == ["return_1d"]
    assert record.model_specs == [{"model_type": "lightgbm", "weight": 1.0, "params": None}]
    assert record.fold_metrics == [{"mae": 0.01, "directional_accuracy": 0.5, "n_test_rows": 100}]
    assert record.holdout_metrics == {"mae": 0.02, "directional_accuracy": 0.55, "n_test_rows": 200}
    assert record.duration_seconds >= 0
    assert record.error is None


def test_a_failed_run_appends_a_minimal_training_run_record(tmp_path):
    run_store = TrainingRunStore(data_dir=tmp_path)

    def failing_train(included_features, model_specs):
        raise ValueError("boom")

    job = TrainingJob(train_fn=failing_train, run_store=run_store)

    job.start()

    _wait_until(lambda: job.status().status == "failed")
    [record] = run_store.read_all()
    assert record.status == "failed"
    assert record.error == "boom"
    assert record.train_tickers is None
    assert record.fold_metrics is None
