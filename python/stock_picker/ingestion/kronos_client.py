"""Bounded JSON-lines transport to the optional, isolated persistent worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import time


class KronosWorkerClient:
    def __init__(self, root: Path, device: str = "mps", timeout: float = 120,
                 command: list[str] | None = None):
        self.root, self.timeout = root, timeout
        self.command = command or [str(root / ".runtime/kronos/venv/bin/python"),
                                   "-m", "tools.kronos.worker", "--device", device]
        self.process: subprocess.Popen | None = None
        self.description: dict | None = None

    def start(self) -> dict:
        self.close()
        # Bazel's hermetic launcher exports Python isolation settings. They
        # belong to the app interpreter, never the independently pinned worker.
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH")}
        environment["PYTHONPATH"] = str(self.root)
        self.process = subprocess.Popen(self.command, cwd=self.root, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None, bufsize=0, env=environment)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            self.description = self.exchange({"describe": True})
            if "execution_fingerprint" not in self.description:
                raise ValueError("worker describe response lacks execution profile")
            return self.description
        except Exception:
            self.close()
            raise

    def exchange(self, payload: dict) -> dict:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Kronos worker is not running")
        try:
            outgoing = memoryview((json.dumps(payload, allow_nan=False) + "\n").encode())
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                deadline = time.monotonic() + self.timeout
                line = bytearray()
                while outgoing or b"\n" not in line:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Kronos worker response timeout")
                    events = selector.select(remaining)
                    if not events:
                        raise TimeoutError("Kronos worker response timeout")
                    for key, mask in events:
                        if mask & selectors.EVENT_WRITE:
                            written = os.write(key.fd, outgoing)
                            outgoing = outgoing[written:]
                            if not outgoing:
                                selector.unregister(self.process.stdin)
                        if mask & selectors.EVENT_READ:
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                raise RuntimeError("Kronos worker exited before a response")
                            line.extend(chunk)
                    if len(line) > 16 * 1024 * 1024:
                        raise ValueError("Kronos response exceeded protocol size limit")
            result = json.loads(line)
            if not isinstance(result, dict) or "error" in result:
                raise ValueError("Kronos worker returned a protocol error")
            return result
        except Exception:
            self.close()
            raise

    def predict(self, requests: list[dict]) -> list[dict]:
        results = self.exchange({"requests": requests}).get("results")
        if not isinstance(results, list) or len(results) != len(requests):
            self.close()
            raise ValueError("Kronos worker returned a different batch length")
        return results

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()
