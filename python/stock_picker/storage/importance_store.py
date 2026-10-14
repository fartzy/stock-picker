"""Durable maximum observed importance, including currently pruned inputs."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


class ImportanceStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.directory = Path(data_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "highest_known.json"
        self.lock_path = self.directory / ".lock"

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "peaks": {}, "artifacts": {}, "issues": {}}
        return json.loads(self.path.read_text())

    def snapshot(self) -> dict:
        with self._locked():
            return self._load()

    def record_many(self, observations: list[tuple[str, dict[str, float]]], *, artifacts=None, issues=None) -> None:
        """Merge maxima under a process lock; publish only complete JSON."""
        with self._locked():
            state = self._load()
            before = json.dumps(state, sort_keys=True)
            for source, values in observations:
                for name, value in values.items():
                    if not math.isfinite(value) or value < 0 or value > 100.000001:
                        raise ValueError(f"invalid normalized importance for {name}: {value}")
                    previous = state["peaks"].get(name)
                    if previous is None or value > previous["value"]:
                        state["peaks"][name] = {"value": float(value), "source": source}
            state["artifacts"].update(artifacts or {})
            state["issues"].update(issues or {})
            if json.dumps(state, sort_keys=True) == before:
                return
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, suffix=".tmp", delete=False) as output:
                    temporary = Path(output.name)
                    json.dump(state, output, indent=2, sort_keys=True)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
