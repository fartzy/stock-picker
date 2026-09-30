"""Fail-closed, research-only checkpoints for the ADR 0021 SVM search.

The manifest identifies the *loaded* training snapshot and experiment code.
SQLite commits each scored SVM block or fold metric atomically, so an
interrupted run can skip completed fits without writing production stores.
Schema v2 stores numeric-only NPZ blocks, never Python objects. Checksums
detect accidental corruption, not authenticity; even a rewritten checksum
cannot enable pickle loading. Legacy pickle checkpoints are not supported.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS

SCHEMA_VERSION = 2
BLOCK_MEMBERS = frozenset({"values", "row_index", "identities", "columns"})
SOURCE_FILES = (
    "requirements.txt",
    "MODULE.bazel",
    ".bazelversion",
    "python/stock_picker/training/svr_stack_search.py",
    "python/stock_picker/training/svr_stack_run.py",
    "python/stock_picker/training/model.py",
    "python/stock_picker/training/dataset.py",
    "python/stock_picker/training/splits.py",
    "python/stock_picker/training/backtest.py",
    "python/stock_picker/features/pruning.py",
    "python/stock_picker/features/stacked_svm.py",
    "python/stock_picker/features/open_pattern_seasonality.py",
    "python/stock_picker/features/structure.py",
    "python/stock_picker/features/weather.py",
    "python/stock_picker/storage/price_store.py",
    "python/stock_picker/storage/feature_store.py",
    "python/stock_picker/storage/universe_store.py",
)
METRIC_KEYS = frozenset({"mae", "acc", "rank_ic", "gated_n", "gated_hit", "gated_avg"})


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_safe(value: object) -> object:
    """Represent unavailable research metrics without nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered loaded values, dtypes and index; reject ambiguous row IDs."""
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError("research snapshot requires unique row indexes")
    if not frame.columns.is_unique:
        raise ValueError("research snapshot requires unique column names")
    if {"ticker", "date"} <= set(frame.columns):
        if frame[["ticker", "date"]].isna().any().any() or frame.duplicated(["ticker", "date"]).any():
            raise ValueError("research snapshot requires unique non-null ticker/date identities")
    digest = hashlib.sha256()
    digest.update(_json_bytes({
        "columns": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes],
        "index_dtype": str(frame.index.dtype), "index_name": frame.index.name, "rows": len(frame),
    }))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _byte_array(value: object) -> np.ndarray:
    """JSON metadata encoded as uint8, so every archive member is numeric."""
    return np.frombuffer(_json_bytes(value), dtype=np.uint8)


def _row_metadata(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    """Exact row/order identities, not an object array or a lossy numeric cast."""
    if not {"ticker", "date"} <= set(scored.columns):
        raise ValueError("research block requires ticker/date identities")
    if scored.index.dtype.kind not in "iu" or not scored["ticker"].map(
        lambda value: isinstance(value, str)
    ).all():
        raise ValueError("research block requires integer row indexes and string tickers")
    return {
        "row_index": scored.index.to_numpy(),
        "identities": _byte_array({
            "tickers": scored["ticker"].tolist(),
            "dates": [pd.Timestamp(value).isoformat() for value in scored["date"]],
        }),
        "columns": _byte_array(list(STACKED_SVM_COLUMNS)),
    }


def _block_context(scored: pd.DataFrame, fit_end: object, score_start: object) -> tuple[str, str, str]:
    """Validate chronology at the persistence boundary as well as in the runner."""
    fingerprint = frame_fingerprint(scored)
    if scored.empty or "date" not in scored:
        raise ValueError("research block requires nonempty dated rows")
    try:
        end, start = pd.Timestamp(fit_end), pd.Timestamp(score_start)
        valid = (
            not pd.isna(end) and not pd.isna(start)
            and start == pd.Timestamp(scored["date"].min()) and end < start
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("research block requires strictly earlier fitting dates") from exc
    if not valid:
        raise ValueError("research block requires strictly earlier fitting dates and exact score start")
    return fingerprint, end.isoformat(), start.isoformat()


def _validate_values(values: np.ndarray, rows: int) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.shape != (rows, len(STACKED_SVM_COLUMNS)) or values.dtype.kind not in "iuf"
    ):
        raise ValueError("research block has invalid numeric output shape or dtype")
    if not np.isfinite(values).all():
        raise ValueError("research block outputs must be finite")


def index_fingerprint(frame: pd.DataFrame) -> str:
    """Identify the precise mask/order within a separately hashed pooled frame."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame.index).to_numpy(dtype="uint64").tobytes()).hexdigest()


def source_identity() -> dict:
    """Record source, dependency/toolchain pins and git state, including edits.

    Bazel's source symlinks resolve back to this checkout. A copied runfiles
    tree must include the provenance files too; never silently omit them.
    """
    root = Path(__file__).resolve().parents[3]
    try:
        source_hashes = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        }
    except OSError as exc:
        raise ValueError("research provenance requires a complete readable source tree") from exc
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *SOURCE_FILES],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Bazel runfiles may not have a .git directory. The source hashes
        # still fingerprint the bytes that this process actually executes.
        head, dirty = None, None
    return {"sha256": source_hashes, "git_head": head, "dirty_sources": dirty}


def dependency_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("numpy", "pandas", "scipy", "lightgbm", "scikit-learn")
    }


def make_manifest(
    pooled: pd.DataFrame,
    train_tickers: list[str],
    holdout: set[str],
    base_features: set[str],
    excluded: set[str],
    candidates: dict[str, tuple[str, ...]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    settings: dict,
    sources: dict | None = None,
    dependencies: dict | None = None,
) -> dict:
    first_train = pooled[splits[0][0]]
    def split_ids(frame: pd.DataFrame, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
        return [
            {"train": index_fingerprint(frame[train]), "test": index_fingerprint(frame[test])}
            for train, test in pairs
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "stacked_columns": list(STACKED_SVM_COLUMNS),
        "pooled_sha256": frame_fingerprint(pooled),
        "rows": len(pooled),
        "train_tickers": train_tickers,
        "holdout_tickers": sorted(holdout),
        "base_features": sorted(base_features),
        "excluded_base_features": sorted(excluded),
        "candidates": {name: list(columns) for name, columns in candidates.items()},
        "folds": split_ids(pooled, splits),
        "seed_folds": split_ids(first_train, seed_splits),
        "settings": settings,
        "source": source_identity() if sources is None else sources,
        "dependencies": dependency_versions() if dependencies is None else dependencies,
    }


class ResearchRun:
    """An immutable manifest plus atomic, validated fold checkpoints."""

    def __init__(self, run_dir: Path, manifest: dict, *, resume: bool):
        self.run_dir = Path(run_dir).resolve()
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("research manifest schema is incompatible; start a new run")
        # Snapshot bytes do not change if the caller later mutates its dictionary.
        self._manifest_bytes = _json_bytes(manifest)
        manifest_path = self.run_dir / "manifest.json"
        db_path = self.run_dir / "progress.sqlite3"
        if resume:
            if not manifest_path.is_file() or not db_path.is_file():
                raise ValueError("resume requires an existing complete research run")
            try:
                saved = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("research manifest is unreadable or partial") from exc
            if not isinstance(saved, dict) or saved.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("research manifest schema is incompatible; start a new run")
            if _json_bytes(saved) != self._manifest_bytes:
                raise ValueError("research manifest mismatch: data, code or experiment config changed")
            self.connection = sqlite3.connect(db_path)
            try:
                self._validate_schema()
            except (ValueError, sqlite3.DatabaseError) as exc:
                self.connection.close()
                raise ValueError("research progress database is corrupted or incompatible") from exc
        else:
            # Refuse to overwrite an existing run; --resume is always explicit.
            self.run_dir.mkdir(parents=True, exist_ok=False)
            manifest_path.write_bytes(self._manifest_bytes)
            self.connection = sqlite3.connect(db_path)
            with self.connection:
                self.connection.execute("CREATE TABLE blocks (name TEXT PRIMARY KEY, input_sha256 TEXT NOT NULL, fit_end TEXT NOT NULL, score_start TEXT NOT NULL, payload_sha256 TEXT NOT NULL, payload BLOB NOT NULL)")
                self.connection.execute("CREATE TABLE metrics (fold INTEGER NOT NULL, name TEXT NOT NULL, subset TEXT NOT NULL, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL, PRIMARY KEY (fold, name))")
                self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _validate_schema(self) -> None:
        version_row = self.connection.execute("PRAGMA user_version").fetchone()
        tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if version_row != (SCHEMA_VERSION,) or tables != {"blocks", "metrics"}:
            raise ValueError("research progress database is incomplete or incompatible")
        expected_columns = {
            "blocks": [
                ("name", "TEXT", 0, 1), ("input_sha256", "TEXT", 1, 0),
                ("fit_end", "TEXT", 1, 0), ("score_start", "TEXT", 1, 0),
                ("payload_sha256", "TEXT", 1, 0), ("payload", "BLOB", 1, 0),
            ],
            "metrics": [
                ("fold", "INTEGER", 1, 1), ("name", "TEXT", 1, 2),
                ("subset", "TEXT", 1, 0), ("payload", "TEXT", 1, 0),
                ("payload_sha256", "TEXT", 1, 0),
            ],
        }
        for table, columns in expected_columns.items():
            # Table names come only from the constants above, never from saved data.
            actual = [
                (row[1], row[2], row[3], row[5])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
            ]
            if actual != columns:
                raise ValueError("research progress database schema is incompatible")
        if self.connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("research progress database is corrupted")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ResearchRun:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def block(
        self, name: str, scored: pd.DataFrame, *, fit_end: object, score_start: object
    ) -> pd.DataFrame | None:
        context = _block_context(scored, fit_end, score_start)
        metadata = _row_metadata(scored)
        row = self.connection.execute(
            "SELECT input_sha256, fit_end, score_start, payload_sha256, payload FROM blocks WHERE name=?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        input_sha, saved_end, saved_start, payload_sha, payload = row
        if (
            (input_sha, saved_end, saved_start) != context or not isinstance(payload, bytes)
            or hashlib.sha256(payload).hexdigest() != payload_sha
        ):
            raise ValueError(f"research block {name} is stale or corrupted")
        try:
            # A bare NPY or pickle is not a block. Object arrays are refused by
            # NumPy before materialization, even if their checksum was rewritten.
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if set(archive.files) != BLOCK_MEMBERS or len(archive.files) != len(BLOCK_MEMBERS):
                    raise ValueError("missing, duplicate or unexpected archive members")
                values = archive["values"]
                _validate_values(values, len(scored))
                for field, expected in metadata.items():
                    actual = archive[field]
                    if (
                        not isinstance(actual, np.ndarray)
                        or actual.dtype != expected.dtype or actual.shape != expected.shape
                        or not np.array_equal(actual, expected)
                    ):
                        raise ValueError(f"invalid {field} or row order")
        except (ValueError, TypeError, EOFError, OSError, KeyError, BadZipFile) as exc:
            raise ValueError(f"research block {name} has an invalid numeric archive: {exc}") from exc
        return scored.assign(**{
            column: values[:, position] for position, column in enumerate(STACKED_SVM_COLUMNS)
        })

    def save_block(
        self, name: str, scored: pd.DataFrame, stacked: pd.DataFrame, *, fit_end: object, score_start: object
    ) -> None:
        if self.block(name, scored, fit_end=fit_end, score_start=score_start) is not None:
            raise ValueError(f"research block {name} already exists")
        if not stacked.index.equals(scored.index) or not stacked.reindex(columns=scored.columns).equals(scored):
            raise ValueError("research block changed scored row identities, labels or base features")
        if list(stacked.columns) != [*scored.columns, *STACKED_SVM_COLUMNS]:
            raise ValueError("research block has missing, reordered or unexpected output columns")
        values = stacked[list(STACKED_SVM_COLUMNS)].to_numpy()
        _validate_values(values, len(scored))
        buffer = io.BytesIO()
        np.savez(buffer, allow_pickle=False, values=values, **_row_metadata(scored))
        payload = buffer.getvalue()
        with self.connection:
            self.connection.execute(
                "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?)",
                (name, *_block_context(scored, fit_end, score_start),
                 hashlib.sha256(payload).hexdigest(), payload),
            )

    def metric(self, fold: int, name: str, subset: tuple[str, ...]) -> dict | None:
        row = self.connection.execute(
            "SELECT subset, payload, payload_sha256 FROM metrics WHERE fold=? AND name=?", (fold, name)
        ).fetchone()
        if row is None:
            return None
        if row[0] != _json_bytes(list(subset)).decode():
            raise ValueError(f"research metric {fold}/{name} belongs to another subset")
        if not isinstance(row[1], str) or hashlib.sha256(row[1].encode()).hexdigest() != row[2]:
            raise ValueError(f"research metric {fold}/{name} is corrupted")
        try:
            metric = json.loads(row[1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"research metric {fold}/{name} is corrupted") from exc
        if not isinstance(metric, dict) or set(metric) != METRIC_KEYS or any(
            value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
            ) for value in metric.values()
        ):
            raise ValueError(f"research metric {fold}/{name} is invalid")
        return {key: (float("nan") if value is None else value) for key, value in metric.items()}

    def save_metric(self, fold: int, name: str, subset: tuple[str, ...], metric: dict) -> None:
        if self.metric(fold, name, subset) is not None:
            raise ValueError(f"research metric {fold}/{name} already exists")
        if set(metric) != METRIC_KEYS:
            raise ValueError("research metric has missing or unexpected fields")
        # rank_ic or gated fields may be NaN for tiny/no-trade test fixtures;
        # JSON stores them as null and restores them for arithmetic below.
        clean = {key: (None if pd.isna(value) else float(value)) for key, value in metric.items()}
        payload = _json_bytes(clean).decode()
        with self.connection:
            self.connection.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                (fold, name, _json_bytes(list(subset)).decode(), payload,
                 hashlib.sha256(payload.encode()).hexdigest()),
            )

    def write_results(self, results: dict) -> None:
        """Atomically publish a human-readable report after all folds finish."""
        report = {
            **results,
            "manifest_sha256": hashlib.sha256(self._manifest_bytes).hexdigest(),
        }
        payload = _json_bytes(_json_safe(report))
        with tempfile.NamedTemporaryFile(dir=self.run_dir, prefix=".results-", delete=False) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.replace(temporary, self.run_dir / "results.json")
        finally:
            temporary.unlink(missing_ok=True)
