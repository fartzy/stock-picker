"""Capture durable runtime evidence without stopping the API or rewriting live stores.

Run from the repo root using .venv/bin/python. SQLite's backup API incorporates
committed WAL contents. Gzip archives preserve the resulting bytes and stay below
GitHub's per-file limit. The capture manifest records both file and logical state.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "docs/archive/local-checkin-2026-11-11"
SNAPSHOT = ROOT / "data/snapshots/2026-11-11"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    changed = subprocess.check_output(["git", "ls-files", "--modified", "-z"], cwd=ROOT).decode().split("\0")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    paths = sorted({name for name in changed + untracked if name.startswith("data/") and not name.startswith("data/snapshots/")})
    validation, row_counts, date_ends = [], Counter(), Counter()
    for count, name in enumerate(paths, 1):
        path = ROOT / name
        before = digest(path)
        record = {"path": name, "bytes": path.stat().st_size, "sha256": before}
        if path.suffix == ".parquet":
            table = pq.read_table(path)
            table.validate(full=True)
            record.update(rows=table.num_rows, columns=table.num_columns)
            row_counts[str(path.relative_to(ROOT).parent)] += table.num_rows
            for field in ("Date", "date", "__index_level_0__"):
                if field in table.column_names and table.num_rows:
                    values = table[field].to_pylist()
                    dates = [str(value) for value in values if value is not None]
                    if dates:
                        record.update(first_date=min(dates), last_date=max(dates))
                        date_ends[name.split("/")[1] + ":" + max(dates)[:10]] += 1
                    break
        elif path.suffix == ".json":
            json.loads(path.read_text())
        if digest(path) != before:
            raise RuntimeError(f"Source changed during validation: {name}; rerun capture")
        validation.append(record)
        if count % 500 == 0:
            print(f"Validated {count}/{len(paths)} changed data files", flush=True)

    snapshots = []
    for relative in ("data/buy_signals", "data/morning_checks", "data/news"):
        for source in sorted((ROOT / relative).rglob("*")):
            if not source.is_file() or source.name.startswith(".") or source.suffix == ".db" or source.name.endswith(("-wal", "-shm")):
                continue
            destination = SNAPSHOT / "files" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            first = digest(source)
            shutil.copy2(source, destination)
            if digest(source) != first or digest(destination) != first:
                raise RuntimeError(f"Source changed during capture: {source}")
            snapshots.append({"source": str(source.relative_to(ROOT)), "archive": str(destination.relative_to(ROOT)),
                              "sha256": first, "captured_at": datetime.now(timezone.utc).isoformat()})

    stores = [source for source in (ROOT / "data").rglob("*.db") if "snapshots" not in source.parts]
    for source in sorted(stores):
        relative = source.relative_to(ROOT)
        with tempfile.TemporaryDirectory(prefix="stock-picker-checkin-") as temporary:
            backup = Path(temporary) / "snapshot.db"
            with sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30) as origin:
                with sqlite3.connect(backup) as target:
                    origin.backup(target)
                    integrity = target.execute("PRAGMA integrity_check").fetchall()
                    if integrity != [("ok",)]:
                        raise RuntimeError(f"SQLite integrity failed: {relative}")
                    tables = [row[0] for row in target.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                    counts = {name: target.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0] for name in tables}
            destination = SNAPSHOT / "sqlite" / relative.with_suffix(".db.gz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with backup.open("rb") as raw, destination.open("wb") as output:
                with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                    shutil.copyfileobj(raw, compressed)
            verifier = hashlib.sha256()
            with gzip.open(destination, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    verifier.update(chunk)
            if verifier.hexdigest() != digest(backup):
                raise RuntimeError("SQLite archive did not round-trip")
            if destination.stat().st_size >= 100 * 1024 * 1024:
                raise RuntimeError("Archive exceeds per-file publication limit")
            snapshots.append({"source": str(relative), "archive": str(destination.relative_to(ROOT)),
                              "sha256": digest(destination), "uncompressed_sha256": verifier.hexdigest(),
                              "bytes": destination.stat().st_size, "tables": counts, "integrity": "ok",
                              "captured_at": datetime.now(timezone.utc).isoformat()})
            print(f"Captured {relative}", flush=True)

    # Only report filenames if a likely credential is found; never emit values.
    patterns = [rb"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}", rb"\bAKIA[A-Z0-9]{16}\b",
                rb"(?i)[\"'](?:api_key|apikey|access_token|secret_access_key)[\"']\s*:\s*[\"'][A-Za-z0-9_./+-]{20,}[\"']"]
    suspect = []
    for name in paths + [row["archive"] for row in snapshots]:
        path = ROOT / name
        if path.suffix in (".json", ".txt", ".md"):
            raw = path.read_bytes()
            if any(re.search(pattern, raw) for pattern in patterns):
                suspect.append(name)
    if suspect:
        write_json(ARCHIVE / "credential-review.json", {"paths": suspect})
        raise RuntimeError("Potential credentials require review; see credential-review.json")

    write_json(ARCHIVE / "data-validation.json", {"started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "files": validation, "row_counts": dict(row_counts), "latest_dates": dict(date_ends),
        "potential_credentials": [], "sqlite_backups": len(stores)})
    write_json(SNAPSHOT / "manifest.json", {"started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "files": snapshots, "excluded": ["credentials", "dependency/build caches", "SQLite WAL/SHM sidecars", "Python bytecode"]})
    print(json.dumps({"validated_data_files": len(validation), "runtime_archives": len(snapshots), "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
