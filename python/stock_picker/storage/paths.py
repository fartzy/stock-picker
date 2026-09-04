"""Resolves the project's data directory root, independent of how a script was
invoked. `bazel run` changes the process's working directory to a runfiles
sandbox with no `data/` in it -- Bazel sets `BUILD_WORKING_DIRECTORY` to the
directory the user actually typed `bazel run` from specifically to solve this;
fall back to the plain working directory for `uv run`/pytest invocations, which
have always been run from the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd())) / "data"
