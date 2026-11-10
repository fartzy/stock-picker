import json
import threading
import time
from types import SimpleNamespace

import pytest

from stock_picker.storage.model_store import ModelStore
from stock_picker.training import intraday_entry_job as module


def test_publish_round_trip_and_failure_preserves_previous_model(tmp_path, monkeypatch):
    store = ModelStore(tmp_path)
    model = SimpleNamespace(metadata={"ready": True, "trained_through": "2026-08-05"})
    monkeypatch.setattr(module, "train_intraday_entry", lambda *args: model)
    phases = []
    builder = lambda: SimpleNamespace(frame=None, metadata={})
    result = module.run_intraday_entry_training(model_store=store, dataset_builder=builder, phase=phases.append)
    assert store.read("intraday_entry").metadata == result
    assert json.loads((tmp_path/"intraday_entry_metadata.json").read_text()) == result
    assert result["training_seconds"] >= 0
    assert len(phases) == 3
    previous = (tmp_path/"intraday_entry.pkl").read_bytes()
    def failed():
        raise ValueError("missing source")
    with pytest.raises(ValueError):
        module.run_intraday_entry_training(model_store=store, dataset_builder=failed)
    assert (tmp_path/"intraday_entry.pkl").read_bytes() == previous


def test_job_does_not_start_second_training_and_finishes():
    entered, release = threading.Event(), threading.Event()
    def training(*, phase):
        phase("Testing")
        entered.set()
        assert release.wait(2)
    job = module.IntradayEntryJob(training)
    assert job.start()
    assert entered.wait(2)
    assert not job.start()
    assert job.status()["phase"] == "Testing"
    release.set()
    deadline = time.monotonic()+2
    while job.status()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(.005)
    assert job.status()["status"] == "completed"
    assert job.status()["completed_at"]
