"""Declarative metadata about each model type `MODEL_TRAINERS` can fit --
version, source location, and package -- surfaced via `/api/model-types` so
the UI's model picker can show "what is this, and where does it come from,"
not just a bare `model_type` string.

`package_version`/`source_file`/`source_line` are resolved live (via
`importlib.metadata`/`inspect`) rather than hand-written, so this never goes
stale as dependencies get upgraded or the file gets edited. Extensible: a
future neural-network or clustering entry is just a new row in the four
dicts below -- though actually *training* a clustering model doesn't fit
today's `TrainedModel`/`Ensemble` continuous-regression-blend contract at
all, so wiring one in for real is separate future work (see README's
roadmap), not something this registry alone solves.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from stock_picker.training.model import MODEL_TRAINERS

# Where in this checkout's own tree a source path lives, so it's found the
# same way whether this process was launched via plain python (an absolute
# path already under the repo) or `bazel run` (a runfiles-sandboxed absolute
# path that still preserves this package-relative suffix).
_PACKAGE_ROOT_MARKER = "python/stock_picker/"


@dataclass
class ModelTypeInfo:
    model_type: str
    display_name: str
    category: str
    package: str
    package_version: str
    source_file: str
    source_line: int
    # None if the origin remote can't be resolved -- the UI falls back to
    # showing source_file:source_line as plain, non-linked text.
    github_url: str | None
    description: str


_DISPLAY_NAMES = {
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}

_CATEGORIES = {
    "lightgbm": "gradient_boosted_trees",
    "random_forest": "tree_ensemble",
    "logistic_regression": "linear_diagnostic",
}

# PyPI distribution name, not the import name -- lightgbm's import (`lightgbm`)
# and dist name happen to match, scikit-learn's (`sklearn`) doesn't.
_PACKAGES = {
    "lightgbm": "lightgbm",
    "random_forest": "scikit-learn",
    "logistic_regression": "scikit-learn",
}

_DESCRIPTIONS = {
    "lightgbm": "Gradient-boosted trees predicting the continuous day-session return. Ensemble member.",
    "random_forest": "Bagged regression trees predicting the continuous day-session return. Ensemble member.",
    "logistic_regression": (
        "Predicts the binarized direction (up/down), not the continuous return -- fit and persisted "
        "standalone as a diagnostic importance lens, never an ensemble member."
    ),
}


def _repo_relative(source_file: str) -> str:
    # rfind, not find: under `bazel run`'s runfiles sandbox the absolute path
    # is itself nested under a directory that also starts with
    # "python/stock_picker/..." (e.g. ".../main.runfiles/_main/python/stock_picker/...")
    # -- the real, innermost package-relative path is always the *last*
    # occurrence of the marker, not the first.
    idx = source_file.rfind(_PACKAGE_ROOT_MARKER)
    return source_file[idx:] if idx != -1 else source_file


def _source_location(trainer) -> tuple[str, int]:
    # All three trainers today are plain, undecorated module-level functions
    # in model.py, so inspect resolves their real location reliably -- this
    # would need re-checking if a future trainer is ever wrapped.
    source_file = inspect.getsourcefile(trainer) or inspect.getfile(trainer)
    _, source_line = inspect.getsourcelines(trainer)
    return _repo_relative(source_file), source_line


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _github_base_url() -> str | None:
    """Best-effort base repo URL from the `origin` remote -- None (never
    raises) if there isn't one, so the UI just falls back to plain text."""
    cwd = os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd())
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return None
    url = result.stdout.strip()
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:") :]
    elif "github.com/" in url:
        path = url.split("github.com/", 1)[1]
    else:
        return None
    return f"https://github.com/{path.removesuffix('.git')}"


def describe_model_types() -> list[ModelTypeInfo]:
    """Every model type MODEL_TRAINERS knows how to fit -- including
    logistic_regression, even though it's excluded from the composable
    ensemble picker (PREDICTIVE_MODEL_TYPES): the user wants to see what
    exists, not just what's currently pickable as an ensemble member."""
    base_url = _github_base_url()
    infos = []
    for model_type, trainer in MODEL_TRAINERS.items():
        source_file, source_line = _source_location(trainer)
        package = _PACKAGES[model_type]
        infos.append(
            ModelTypeInfo(
                model_type=model_type,
                display_name=_DISPLAY_NAMES[model_type],
                category=_CATEGORIES[model_type],
                package=package,
                package_version=_package_version(package),
                source_file=source_file,
                source_line=source_line,
                github_url=f"{base_url}/blob/main/{source_file}#L{source_line}" if base_url else None,
                description=_DESCRIPTIONS[model_type],
            )
        )
    return infos
