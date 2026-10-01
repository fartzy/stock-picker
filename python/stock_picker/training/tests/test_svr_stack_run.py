"""Adversarial format and durability checks for research-only checkpoints."""

import hashlib
import io
import json
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training import svr_stack_run as checkpoint
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.svr_stack_run import ResearchRun, SCHEMA_VERSION

DATES = {"fit_end": "2026-01-07", "score_start": "2026-01-08"}
MANIFEST = {"schema_version": SCHEMA_VERSION, "test": "numeric-checkpoints"}
METRIC = {"mae": 0.01, "acc": 0.5, "rank_ic": float("nan"),
          "gated_n": 0, "gated_hit": float("nan"), "gated_avg": float("nan")}


@pytest.fixture
def scored():
    return pd.DataFrame({
        "ticker": ["BBB", "AAA"],
        "date": pd.to_datetime(["2026-01-08", "2026-01-08"]),
        "signal": [np.nan, 0.2], LABEL_COLUMN: [0.01, -0.01],
    }, index=[11, 10])


@pytest.fixture
def run(tmp_path):
    with ResearchRun(tmp_path / "run", MANIFEST, resume=False) as current:
        yield current


def _stacked(scored, dtype=np.float64):
    return scored.assign(**{
        name: np.asarray([0.1 + position, -0.1 - position], dtype=dtype)
        for position, name in enumerate(STACKED_SVM_COLUMNS)
    })


def _members(run):
    payload = run.connection.execute("SELECT payload FROM blocks").fetchone()[0]
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _archive(members, *, allow_pickle=False):
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=allow_pickle, **members)
    return buffer.getvalue()


def _replace_payload(run, payload):
    # A colocated checksum is not authentication. Attack/malformed cases must
    # still fail when an attacker can replace both the bytes and their digest.
    with run.connection:
        run.connection.execute(
            "UPDATE blocks SET payload=?, payload_sha256=? WHERE name='outer-1'",
            (payload, hashlib.sha256(payload).hexdigest()),
        )


def test_scipy_runtime_version_change_rejects_resume(tmp_path, monkeypatch):
    versions = {name: "unchanged" for name in ("numpy", "pandas", "lightgbm", "scikit-learn")}
    versions["scipy"] = "before"
    monkeypatch.setattr(checkpoint, "version", versions.__getitem__)
    before = checkpoint.dependency_versions()
    assert before == versions
    run_dir = tmp_path / "run"
    with ResearchRun(run_dir, {**MANIFEST, "dependencies": before}, resume=False):
        pass
    versions["scipy"] = "after"
    after = checkpoint.dependency_versions()
    assert {name for name in before if before[name] != after[name]} == {"scipy"}
    with pytest.raises(ValueError, match="manifest mismatch"):
        ResearchRun(run_dir, {**MANIFEST, "dependencies": after}, resume=True)


@pytest.fixture
def provenance_tree(tmp_path, monkeypatch):
    root = tmp_path / "source-tree"
    for name in checkpoint.SOURCE_FILES:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(name + "\n")
    monkeypatch.setattr(checkpoint, "__file__", str(root / "python/stock_picker/training/svr_stack_run.py"))
    # Keep git identity constant: the actual file bytes must detect this edit.
    monkeypatch.setattr(checkpoint.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="unchanged"))
    return root


@pytest.mark.parametrize("name", ["requirements.txt", "MODULE.bazel", ".bazelversion"])
def test_dependency_and_toolchain_source_edits_reject_resume(tmp_path, provenance_tree, name):
    before = checkpoint.source_identity()
    assert name in before["sha256"]
    source = provenance_tree / name
    assert before["sha256"][name] == hashlib.sha256(source.read_bytes()).hexdigest()
    run_dir = tmp_path / "run"
    with ResearchRun(run_dir, {**MANIFEST, "source": before}, resume=False):
        pass
    source.write_text("changed dependency or toolchain pin\n")
    after = checkpoint.source_identity()
    assert before["git_head"] == after["git_head"]
    assert before["dirty_sources"] == after["dirty_sources"]
    assert {key for key in before["sha256"] if before["sha256"][key] != after["sha256"][key]} == {name}
    with pytest.raises(ValueError, match="manifest mismatch"):
        ResearchRun(run_dir, {**MANIFEST, "source": after}, resume=True)


def test_missing_dependency_provenance_fails_closed(provenance_tree):
    (provenance_tree / "requirements.txt").unlink()
    with pytest.raises(ValueError, match="complete readable source tree"):
        checkpoint.source_identity()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numeric_roundtrip_preserves_dtype_row_order_and_base_nans(run, scored, dtype):
    stacked = _stacked(scored, dtype)
    run.save_block("outer-1", scored, stacked, **DATES)
    assert run.connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    assert json.loads((run.run_dir / "manifest.json").read_text())["schema_version"] == SCHEMA_VERSION
    members = _members(run)
    assert set(members) == {"values", "row_index", "identities", "columns"}
    assert all(value.dtype.kind in "iuf" for value in members.values())
    assert members["values"].dtype == dtype
    assert members["row_index"].tolist() == [11, 10]
    assert json.loads(members["columns"].tobytes()) == list(STACKED_SVM_COLUMNS)
    assert json.loads(members["identities"].tobytes())["tickers"] == ["BBB", "AAA"]
    with ResearchRun(run.run_dir, MANIFEST, resume=True) as resumed:
        pd.testing.assert_frame_equal(resumed.block("outer-1", scored, **DATES), stacked)


@pytest.mark.parametrize("malformation", [
    "missing_member", "extra_member", "rows", "columns_count", "flat_values",
    "object", "text", "bool", "complex", "nan", "inf", "minus_inf",
    "column_names", "row_order", "row_dtype", "identities", "metadata_dtype",
])
def test_invalid_archives_fail_even_with_rewritten_checksum(run, scored, malformation):
    run.save_block("outer-1", scored, _stacked(scored), **DATES)
    members = _members(run)
    if malformation == "missing_member":
        del members["columns"]
    elif malformation == "extra_member":
        members["extra"] = np.zeros(1)
    elif malformation == "rows":
        members["values"] = members["values"][:1]
    elif malformation == "columns_count":
        members["values"] = members["values"][:, :6]
    elif malformation == "flat_values":
        members["values"] = members["values"].ravel()
    elif malformation in {"object", "text", "bool", "complex"}:
        dtype = {"object": object, "text": str, "bool": bool, "complex": complex}[malformation]
        members["values"] = members["values"].astype(dtype)
    elif malformation in {"nan", "inf", "minus_inf"}:
        members["values"][0, 0] = {"nan": np.nan, "inf": np.inf, "minus_inf": -np.inf}[malformation]
    elif malformation == "column_names":
        members["columns"] = np.frombuffer(json.dumps(list(reversed(STACKED_SVM_COLUMNS))).encode(), dtype=np.uint8)
    elif malformation == "row_order":
        members["row_index"] = members["row_index"][::-1]
    elif malformation == "row_dtype":
        members["row_index"] = members["row_index"].astype(float)
    elif malformation == "identities":
        members["identities"] = members["identities"][::-1]
    elif malformation == "metadata_dtype":
        members["columns"] = members["columns"].astype(np.int64)
    _replace_payload(run, _archive(members, allow_pickle=malformation == "object"))
    with pytest.raises(ValueError, match="invalid numeric archive"):
        run.block("outer-1", scored, **DATES)


@pytest.mark.parametrize("payload_kind", ["pickle", "object_npz", "npy", "truncated_npz", "garbage"])
def test_wrong_format_or_executable_payload_is_never_executed(run, scored, tmp_path, payload_kind):
    marker = tmp_path / "must-not-exist"

    class ExecutablePayload:
        def __reduce__(self):
            return Path.write_text, (marker, "UNSAFE")

    run.save_block("outer-1", scored, _stacked(scored), **DATES)
    if payload_kind == "pickle":
        payload = pickle.dumps(ExecutablePayload())
    elif payload_kind == "object_npz":
        members = _members(run)
        members["values"] = np.full((len(scored), len(STACKED_SVM_COLUMNS)), ExecutablePayload(), dtype=object)
        payload = _archive(members, allow_pickle=True)
    elif payload_kind == "npy":
        buffer = io.BytesIO()
        np.save(buffer, np.zeros((len(scored), len(STACKED_SVM_COLUMNS))), allow_pickle=False)
        payload = buffer.getvalue()
    elif payload_kind == "truncated_npz":
        payload = _archive(_members(run))[:-24]
    else:
        payload = b"not an archive"
    _replace_payload(run, payload)
    with pytest.raises(ValueError, match="invalid numeric archive"):
        run.block("outer-1", scored, **DATES)
    assert not marker.exists()


@pytest.mark.parametrize("field", ["values", "columns"])
def test_zip_member_without_npy_header_is_rejected(run, scored, field):
    run.save_block("outer-1", scored, _stacked(scored), **DATES)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, value in _members(run).items():
            member = io.BytesIO()
            np.save(member, value, allow_pickle=False)
            archive.writestr(name + ".npy", b"not a numeric array" if name == field else member.getvalue())
    _replace_payload(run, buffer.getvalue())
    with pytest.raises(ValueError, match="invalid numeric archive"):
        run.block("outer-1", scored, **DATES)


def test_duplicate_archive_members_are_rejected(run, scored):
    run.save_block("outer-1", scored, _stacked(scored), **DATES)
    buffer = io.BytesIO(_archive(_members(run)))
    with ZipFile(buffer, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
        archive.writestr("values.npy", b"duplicate")
    _replace_payload(run, buffer.getvalue())
    with pytest.raises(ValueError, match="invalid numeric archive"):
        run.block("outer-1", scored, **DATES)


@pytest.mark.parametrize("change", ["base", "label", "index", "missing", "extra", "reordered", "nan", "inf", "object"])
def test_save_refuses_changed_base_rows_or_invalid_outputs(run, scored, change):
    stacked = _stacked(scored)
    if change == "base":
        stacked["signal"] = 99.0
    elif change == "label":
        stacked[LABEL_COLUMN] = 99.0
    elif change == "index":
        stacked = stacked.iloc[::-1]
    elif change == "missing":
        stacked = stacked.drop(columns=STACKED_SVM_COLUMNS[0])
    elif change == "extra":
        stacked["unexpected"] = 1.0
    elif change == "reordered":
        stacked = stacked[[*scored.columns, *reversed(STACKED_SVM_COLUMNS)]]
    elif change == "object":
        stacked[STACKED_SVM_COLUMNS[0]] = stacked[STACKED_SVM_COLUMNS[0]].astype(object)
    else:
        stacked[STACKED_SVM_COLUMNS[0]] = np.nan if change == "nan" else np.inf
    with pytest.raises(ValueError, match="research block"):
        run.save_block("outer-1", scored, stacked, **DATES)
    assert run.connection.execute("SELECT COUNT(*) FROM blocks").fetchone() == (0,)


@pytest.mark.parametrize("dates", [
    {"fit_end": "2026-01-08", "score_start": "2026-01-08"},
    {"fit_end": "2026-01-09", "score_start": "2026-01-08"},
    {"fit_end": "2026-01-07", "score_start": "2026-01-09"},
    {"fit_end": "NaT", "score_start": "2026-01-08"},
])
def test_persistence_boundary_requires_earlier_dates_and_exact_score_start(run, scored, dates):
    with pytest.raises(ValueError, match="strictly earlier"):
        run.save_block("outer-1", scored, _stacked(scored), **dates)
    with pytest.raises(ValueError, match="strictly earlier"):
        run.block("outer-1", scored, **dates)


@pytest.mark.parametrize("legacy", ["manifest", "sqlite"])
def test_legacy_versions_are_rejected_without_conversion(tmp_path, legacy):
    run_dir = tmp_path / "run"
    with ResearchRun(run_dir, MANIFEST, resume=False) as run:
        if legacy == "sqlite":
            run.connection.execute("PRAGMA user_version = 1")
    if legacy == "manifest":
        (run_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "schema_version": 1}))
    with pytest.raises(ValueError, match="incompatible"):
        ResearchRun(run_dir, MANIFEST, resume=True)
    with pytest.raises(ValueError, match="incompatible"):
        ResearchRun(tmp_path / "other", {"schema_version": 1}, resume=False)
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("corruption", ["missing_manifest", "partial_manifest", "wrong_schema", "invalid_database"])
def test_partial_or_corrupt_run_cannot_resume(tmp_path, corruption):
    run_dir = tmp_path / "run"
    with ResearchRun(run_dir, MANIFEST, resume=False) as run:
        if corruption == "wrong_schema":
            run.connection.execute("DROP TABLE blocks")
            run.connection.execute(
                "CREATE TABLE blocks (name, input_sha256, fit_end, score_start, payload_sha256, payload)"
            )
    if corruption == "missing_manifest":
        (run_dir / "manifest.json").unlink()
    elif corruption == "partial_manifest":
        (run_dir / "manifest.json").write_text('{"schema_version":')
    elif corruption == "invalid_database":
        (run_dir / "progress.sqlite3").write_bytes(b"invalid database")
    with pytest.raises(ValueError, match="complete|partial|corrupted|incompatible"):
        ResearchRun(run_dir, MANIFEST, resume=True)


def test_failed_inserts_leave_no_partial_blocks_or_metrics(run, scored):
    for table in ("blocks", "metrics"):
        run.connection.execute(
            f"CREATE TRIGGER stop_insert BEFORE INSERT ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="simulated interruption"):
            if table == "blocks":
                run.save_block("outer-1", scored, _stacked(scored), **DATES)
            else:
                run.save_metric(1, "baseline", (), METRIC)
        assert run.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
        run.connection.execute("DROP TRIGGER stop_insert")
    run.save_block("outer-1", scored, _stacked(scored), **DATES)
    run.save_metric(1, "baseline", (), METRIC)
    with pytest.raises(ValueError, match="already exists"):
        run.save_block("outer-1", scored, _stacked(scored), **DATES)
    with pytest.raises(ValueError, match="already exists"):
        run.save_metric(1, "baseline", (), METRIC)


def test_results_are_deterministic_and_replacement_failure_preserves_old_report(run, monkeypatch):
    run.write_results({"baseline": [METRIC], "candidates": {}})
    before = (run.run_dir / "results.json").read_bytes()
    assert b"NaN" not in before and json.loads(before)["baseline"][0]["rank_ic"] is None
    run.write_results({"candidates": {}, "baseline": [METRIC]})
    assert (run.run_dir / "results.json").read_bytes() == before

    def interrupt_replace(*args):
        raise OSError("simulated result publication interruption")

    monkeypatch.setattr(checkpoint.os, "replace", interrupt_replace)
    with pytest.raises(OSError, match="publication interruption"):
        run.write_results({"baseline": [], "candidates": {}})
    assert (run.run_dir / "results.json").read_bytes() == before
    assert not list(run.run_dir.glob(".results-*"))
