"""Isolated, bounded training job; never invokes morning Rank/Fit training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import logging
import os
import tempfile
import threading
import time
from typing import Callable

from stock_picker.storage.model_store import ModelStore
from stock_picker.training.intraday_entry_dataset import build_intraday_entry_dataset
from stock_picker.training.intraday_entry_model import MODEL_NAME, train_intraday_entry


def run_intraday_entry_training(*, model_store=None, dataset_builder=build_intraday_entry_dataset,
                               phase: Callable[[str], None] | None = None) -> dict:
    started = time.monotonic()
    store = model_store if model_store is not None else ModelStore()
    report = phase if phase is not None else lambda _: None
    report("Loading saved historical minutes")
    dataset = dataset_builder()
    report("Fitting and evaluating the separate entry model")
    model = train_intraday_entry(dataset.frame, dataset.metadata)
    report("Saving evaluated model")
    model.metadata["training_seconds"] = round(time.monotonic() - started, 3)
    serialized = json.dumps(model.metadata, indent=2, allow_nan=False) + "\n"
    # Small human-readable evidence alongside the pickle, atomically published.
    destination = store.data_dir / f"{MODEL_NAME}_metadata.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=store.data_dir, prefix=".entry-", delete=False) as f:
            temporary = f.name
            f.write(serialized)
        store.write(MODEL_NAME, model)
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return model.metadata


@dataclass
class IntradayEntryJobStatus:
    status: str = "idle"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    phase: str | None = None


class IntradayEntryJob:
    def __init__(self, train_fn=run_intraday_entry_training):
        self._lock = threading.Lock()
        self._state = IntradayEntryJobStatus()
        self._train_fn = train_fn

    def status(self) -> dict:
        with self._lock:
            return asdict(self._state)

    def start(self) -> bool:
        with self._lock:
            if self._state.status == "running":
                return False
            self._state = IntradayEntryJobStatus("running", datetime.now(timezone.utc).isoformat(),
                                                phase="Starting entry training")

        def phase(value):
            with self._lock:
                self._state.phase = value

        def run():
            try:
                self._train_fn(phase=phase)
            except Exception:
                logging.getLogger(__name__).exception("Intraday Entry training failed")
                with self._lock:
                    self._state = replace(self._state, status="failed", phase=None,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        error="Entry training failed; check that the frozen historical minute dataset is available.")
            else:
                with self._lock:
                    self._state = replace(self._state, status="completed", phase=None,
                        completed_at=datetime.now(timezone.utc).isoformat())
        threading.Thread(target=run, name="intraday-entry-training", daemon=True).start()
        return True


intraday_entry_job = IntradayEntryJob()
