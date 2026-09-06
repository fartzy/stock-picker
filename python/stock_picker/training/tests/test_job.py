import threading
import time

from stock_picker.training.job import TrainingJob


def _wait_until(predicate, timeout=2.0):
    """Polls until predicate() is true or the timeout elapses -- avoids a
    flaky fixed sleep while waiting for the background thread to progress."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "timed out waiting for condition"


def test_status_starts_idle():
    job = TrainingJob(train_fn=lambda included_features, model_specs: None)

    assert job.status().status == "idle"


def test_start_runs_train_fn_and_reports_completed():
    job = TrainingJob(train_fn=lambda included_features, model_specs: "fake summary")

    started = job.start()

    assert started is True
    _wait_until(lambda: job.status().status == "completed")
    status = job.status()
    assert status.result == "fake summary"
    assert status.error is None
    assert status.started_at is not None
    assert status.completed_at is not None


def test_start_passes_included_features_through_to_train_fn():
    received = {}

    def fake_train(included_features, model_specs):
        received["included_features"] = included_features
        return None

    job = TrainingJob(train_fn=fake_train)
    job.start(included_features={"return_1d"})

    _wait_until(lambda: job.status().status == "completed")
    assert received["included_features"] == {"return_1d"}


def test_start_passes_model_specs_through_to_train_fn():
    received = {}

    def fake_train(included_features, model_specs):
        received["model_specs"] = model_specs
        return None

    job = TrainingJob(train_fn=fake_train)
    job.start(model_specs=["fake spec"])

    _wait_until(lambda: job.status().status == "completed")
    assert received["model_specs"] == ["fake spec"]


def test_start_returns_false_while_a_run_is_already_in_progress():
    release = threading.Event()
    job = TrainingJob(train_fn=lambda included_features, model_specs: release.wait(timeout=2.0))

    first_started = job.start()
    _wait_until(lambda: job.status().status == "running")
    second_started = job.start()

    assert first_started is True
    assert second_started is False

    release.set()
    _wait_until(lambda: job.status().status == "completed")


def test_a_train_fn_exception_reports_failed_with_the_error_message():
    def failing_train(included_features, model_specs):
        raise ValueError("boom")

    job = TrainingJob(train_fn=failing_train)
    job.start()

    _wait_until(lambda: job.status().status == "failed")
    status = job.status()
    assert status.error == "boom"
    assert status.result is None
