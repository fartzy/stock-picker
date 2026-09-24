"""Persisted Test run snapshots. In-memory job state is gone after restart."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from stock_picker.storage.paths import data_root

KEEP = 20


def _dir(data_dir: Path | None = None) -> Path:
    path = (data_dir or data_root()) / "morning_checks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_stamp(value: str) -> str:
    return value.replace(":", "").replace("+", "p")[:32]


def save(payload: dict, data_dir: Path | None = None) -> Path:
    folder = _dir(data_dir)
    stamp = _safe_stamp(payload.get("started_at") or datetime.now().astimezone().isoformat())
    path = folder / f"{stamp}.json"
    path.write_text(json.dumps(payload, default=str))
    prune(folder)
    return path


def prune(folder: Path, keep: int = KEEP) -> None:
    files = sorted(folder.glob("*.json"), key=lambda p: p.name, reverse=True)
    for extra in files[keep:]:
        extra.unlink(missing_ok=True)


def list_runs(data_dir: Path | None = None) -> list[dict]:
    folder = _dir(data_dir)
    out: list[dict] = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.name, reverse=True):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "id": path.stem,
                "started_at": payload.get("started_at"),
                "completed_at": payload.get("completed_at"),
                "which": payload.get("which"),
                "status": payload.get("status"),
                "n_quotes": payload.get("n_quotes"),
            }
        )
    return out


def read(run_id: str, data_dir: Path | None = None) -> dict | None:
    path = _dir(data_dir) / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def latest(data_dir: Path | None = None) -> dict | None:
    runs = list_runs(data_dir)
    if not runs:
        return None
    return read(runs[0]["id"], data_dir)
