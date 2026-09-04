from pathlib import Path

from stock_picker.storage.paths import data_root


def test_data_root_uses_build_working_directory_when_set(monkeypatch):
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", "/somewhere/repo")

    assert data_root() == Path("/somewhere/repo/data")


def test_data_root_falls_back_to_cwd_without_bazel_run(monkeypatch):
    monkeypatch.delenv("BUILD_WORKING_DIRECTORY", raising=False)

    assert data_root() == Path.cwd() / "data"
