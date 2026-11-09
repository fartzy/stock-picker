"""Stop only the captured obsolete nightly, after the orchestrator authorizes it.

Without --execute this is a read-only identity/model check. API/UI processes
are never signalled. A changed PID identity or serving-model hash aborts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def process_identity(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,lstart=,command="],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    preflight = json.loads((HERE / "deployment_preflight.json").read_text())
    expected = preflight["expected_nightly"]
    pid = expected["pid"]
    if pid in (preflight["api_pid"], preflight["ui_pid"]):
        raise RuntimeError("Refusing to signal an API/UI process")
    identity = process_identity(pid)
    fields = identity.split(None, 7)
    if len(fields) < 8 or int(fields[0]) != pid or int(fields[1]) != expected["ppid"]:
        raise RuntimeError(f"Nightly PID or parent identity changed: {identity!r}")
    if " ".join(fields[2:7]) != " ".join(expected["started"].split()):
        raise RuntimeError("Nightly start time changed")
    if not fields[7].endswith(expected["command_suffix"]):
        raise RuntimeError("Nightly executable identity changed")
    for relative, record in preflight["models"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Serving model changed since preflight: {relative}")
    print(json.dumps({"identity": identity, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while process_identity(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if process_identity(pid):
        raise RuntimeError("SIGTERM did not stop the captured nightly; no stronger signal sent")
    after = {}
    for relative, record in preflight["models"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        after[relative] = observed
        if observed != record["sha256"]:
            raise RuntimeError(f"Serving model changed during termination: {relative}")
    result = {
        "terminated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": pid, "captured_identity": identity, "signal": "SIGTERM",
        "model_hashes_after": after, "api_ui_signalled": False,
    }
    (HERE / "nightly_superseded.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
