"""Single in-memory training-job state -- deliberately not a real job queue;
this is a local single-user dev tool, so "is a run in progress, and what did
the last one produce" is all that's needed. State doesn't survive a server
restart, which is the right default: a fresh server means no run is in
flight.

A class (not bare module globals) so tests can construct their own instance
with a fake `train_fn` instead of actually running walk-forward training.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from stock_picker.training.main import TrainingSummary, run_training

Status = Literal["idle", "running", "completed", "failed"]


@dataclass
class JobStatus:
    status: Status = "idle"
    started_at: str | None = None
    completed_at: str | None = None
    result: TrainingSummary | None = None
    error: str | None = None


class TrainingJob:
    def __init__(self, train_fn=run_training) -> None:
        self._lock = threading.Lock()
        self._train_fn = train_fn
        self._state = JobStatus()

    def status(self) -> JobStatus:
        with self._lock:
            return replace(self._state)

    def start(self, included_features: set[str] | None = None) -> bool:
        """Starts a training run on a background thread. Returns False (and
        starts nothing) if a run is already in progress."""
        with self._lock:
            if self._state.status == "running":
                return False
            self._state = JobStatus(status="running", started_at=_now())

        def _run() -> None:
            try:
                result = self._train_fn(included_features=included_features)
                with self._lock:
                    self._state = replace(self._state, status="completed", completed_at=_now(), result=result)
            except Exception as exc:  # noqa: BLE001 -- surfaced via /api/training/status, not swallowed
                with self._lock:
                    self._state = replace(self._state, status="failed", completed_at=_now(), error=str(exc))

        threading.Thread(target=_run, daemon=True).start()
        return True


def _now() -> str:
    return datetime.now().astimezone().isoformat()


_default_job = TrainingJob()


def status() -> JobStatus:
    return _default_job.status()


def start(included_features: set[str] | None = None) -> bool:
    return _default_job.start(included_features)
