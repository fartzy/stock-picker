"""In-memory status for a live morning scan (manual or launchd).

Same shape as training/job.py -- local single-user, does not survive restart.
"""

from __future__ import annotations

import threading
from dataclasses import replace

from stock_picker.training.job import JobStatus, _now
from stock_picker.training.morning import run_morning


class MorningScanJob:
    def __init__(self, run_fn=run_morning) -> None:
        self._lock = threading.Lock()
        self._run_fn = run_fn
        self._state = JobStatus()

    def status(self) -> JobStatus:
        with self._lock:
            return replace(self._state)

    def start(self) -> bool:
        with self._lock:
            if self._state.status == "running":
                return False
            started_at = _now()
            self._state = JobStatus(status="running", started_at=started_at)

        def _run() -> None:
            try:
                code = self._run_fn(ignore_disabled=True)
                with self._lock:
                    self._state = JobStatus(
                        status="completed" if code == 0 else "failed",
                        started_at=started_at,
                        completed_at=_now(),
                        error=None if code == 0 else f"morning exited {code}",
                    )
            except Exception as exc:
                with self._lock:
                    self._state = JobStatus(
                        status="failed",
                        started_at=started_at,
                        completed_at=_now(),
                        error=str(exc),
                    )

        threading.Thread(target=_run, daemon=True).start()
        return True


_default = MorningScanJob()


def status() -> JobStatus:
    return _default.status()


def start() -> bool:
    return _default.start()
