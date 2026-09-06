"""Single in-memory training-job state -- deliberately not a real job queue;
this is a local single-user dev tool, so "is a run in progress, and what did
the last one produce" is all that's needed. State doesn't survive a server
restart, which is the right default: a fresh server means no run is in
flight.

A class (not bare module globals) so tests can construct their own instance
with a fake `train_fn` instead of actually running walk-forward training.

Each completed/failed run is also appended to `storage/training_run_store.py`
(a separate, persisted history) so "what happened across past runs" survives
a restart even though this in-memory job status doesn't.
"""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from stock_picker.storage.training_run_store import TrainingRunRecord, TrainingRunStore
from stock_picker.training.ensemble import ModelSpec
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
    def __init__(self, train_fn=run_training, run_store: TrainingRunStore | None = None) -> None:
        self._lock = threading.Lock()
        self._train_fn = train_fn
        self._run_store = run_store if run_store is not None else TrainingRunStore()
        self._state = JobStatus()

    def status(self) -> JobStatus:
        with self._lock:
            return replace(self._state)

    def start(
        self, included_features: set[str] | None = None, model_specs: list[ModelSpec] | None = None
    ) -> bool:
        """Starts a training run on a background thread. Returns False (and
        starts nothing) if a run is already in progress."""
        with self._lock:
            if self._state.status == "running":
                return False
            started_at = _now()
            self._state = JobStatus(status="running", started_at=started_at)

        def _run() -> None:
            # Persist the run record *before* flipping self._state, in both
            # branches -- otherwise a caller polling /api/training/status and
            # then immediately hitting /api/training/runs could see
            # "completed"/"failed" a moment before the history reflects it.
            run_id = uuid.uuid4().hex
            try:
                result = self._train_fn(included_features=included_features, model_specs=model_specs)
                completed_at = _now()
                self._run_store.append(
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
                with self._lock:
                    self._state = replace(self._state, status="completed", completed_at=completed_at, result=result)
            except Exception as exc:  # noqa: BLE001 -- surfaced via /api/training/status, not swallowed
                completed_at = _now()
                # No tickers/features/specs here -- run_training() prints them
                # immediately after computing them (before the calls that can
                # raise), so a failure's provenance is still in the server
                # log even though it can't make it into this record.
                self._run_store.append(
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
                with self._lock:
                    self._state = replace(self._state, status="failed", completed_at=completed_at, error=str(exc))

        threading.Thread(target=_run, daemon=True).start()
        return True


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _duration_seconds(started_at: str, completed_at: str) -> float:
    return (datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds()


def _git_commit() -> str | None:
    """Best-effort HEAD commit for a run's provenance -- None (never raises)
    if git isn't on PATH or this isn't a checkout. cwd mirrors
    storage/paths.py's data_root() reasoning: `bazel run` sandboxes the
    process's actual cwd to a runfiles dir with no `.git` in it, but sets
    `BUILD_WORKING_DIRECTORY` to the directory the user invoked bazel from.
    """
    cwd = os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return None


_default_job = TrainingJob()


def status() -> JobStatus:
    return _default_job.status()


def start(included_features: set[str] | None = None, model_specs: list[ModelSpec] | None = None) -> bool:
    return _default_job.start(included_features, model_specs)
