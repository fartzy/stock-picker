"""Install the fixed inference runtime in .runtime/kronos, never the app venv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from tools.kronos.contract import read_manifest, verify_assets

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent


def download(url: str, path: Path, expected: str) -> None:
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
        temporary.unlink()
        raise ValueError(f"checksum mismatch for {url}")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".runtime" / "kronos")
    parser.add_argument("--compile-lock", action="store_true", help="explicitly regenerate dependency lock")
    args = parser.parse_args()
    manifest = read_manifest()
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, UV_CACHE_DIR=str(runtime / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(runtime / "python"))
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required; install it before setting up this optional runtime")
    python = runtime / "venv" / "bin" / "python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", manifest["python"], str(runtime / "venv")], env=environment, check=True)
    actual_python = subprocess.check_output([str(python), "-c", "import platform; print(platform.python_version())"], text=True).strip()
    if actual_python != manifest["python"]:
        raise RuntimeError(f"runtime Python {actual_python} differs from pinned {manifest['python']}; use a new --runtime directory")
    lock = CONFIG / "requirements.lock"
    if args.compile_lock:
        subprocess.run([uv, "pip", "compile", "tools/kronos/requirements.in", "--python-version", manifest["python"], "--generate-hashes", "--output-file", "tools/kronos/requirements.lock"], cwd=ROOT, env=environment, check=True)
    if not lock.exists():
        raise RuntimeError("requirements.lock is missing; maintainers must generate it with --compile-lock")
    subprocess.run([uv, "pip", "sync", "--python", str(python), "--require-hashes", str(lock)], env=environment, check=True)
    for kind in ("source", "model", "tokenizer"):
        item = manifest[kind]
        for name, expected in item["files"].items():
            if kind == "source":
                url = f"https://raw.githubusercontent.com/{item['repository']}/{item['revision']}/{name}"
            else:
                url = f"https://huggingface.co/{item['repository']}/resolve/{item['revision']}/{name}"
            download(url, runtime / kind / name, expected)
    verify_assets(runtime, manifest)
    print(json.dumps({"runtime": str(runtime), "python": str(python), "assets_verified": True}))


if __name__ == "__main__":
    main()
