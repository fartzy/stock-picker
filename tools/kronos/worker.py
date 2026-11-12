"""Offline JSON-lines worker; model weights stay loaded between requests.

    .runtime/kronos/venv/bin/python -m tools.kronos.worker --device cpu

Each input line is {"requests": [request, ...]}. Each result preserves raw
forecasts and validation status. An invalid forecast is never silently repaired.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

from tools.kronos.contract import FLOAT32_MAX, digest, manifest_digest, prepare_request, read_manifest, validate_ohlcv, verify_assets

ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DailyWorker:
    def __init__(self, runtime: Path, device: str = "cpu"):
        started = time.perf_counter()
        self.manifest = read_manifest()
        verify_assets(runtime, self.manifest)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        sys.path.insert(0, str(runtime / "source"))
        import exchange_calendars as xcals
        import numpy as np
        import pandas as pd
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        self.np, self.pd, self.torch = np, pd, torch
        self.calendar = xcals.get_calendar("XNYS")
        # Small forecasts should not monopolize the API's host CPU.
        torch.set_num_threads(2)
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self.device = device
        # Official API: huggingface_hub v0.33.1 package_reference/mixins.
        with redirect_stdout(sys.stderr):
            tokenizer = KronosTokenizer.from_pretrained(str(runtime / "tokenizer"), local_files_only=True)
            model = Kronos.from_pretrained(str(runtime / "model"), local_files_only=True)
        model.eval()
        tokenizer.eval()
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=self.manifest["parameters"]["max_context"])
        self.versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "huggingface_hub", "exchange_calendars", "einops", "safetensors", "pyarrow")}
        self.execution_profile = {
            "device": device, "python": platform.python_version(),
            "platform": sys.platform, "architecture": platform.machine(),
            "packages": self.versions,
            "files": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("worker.py", "contract.py", "requirements.lock")},
        }
        self.execution_fingerprint = digest(self.execution_profile)
        self.load_seconds = time.perf_counter() - started

    def predict(self, request: dict) -> dict:
        prepared = prepare_request(request, self.manifest, self.execution_fingerprint)
        context = self.pd.DataFrame(prepared["bars"])
        sessions = self.pd.DatetimeIndex(context["session"])
        expected = self.calendar.sessions_in_range(sessions[0], sessions[-1])
        if not sessions.equals(expected):
            raise ValueError("history must contain consecutive XNYS sessions without gaps")
        if self.calendar.next_session(sessions[-1]).date().isoformat() != prepared["target_session"]:
            raise ValueError("target must be the next XNYS session after context")
        # This is a conservative calendar buffer, not a provider observation
        # timestamp. Stored histories do not provide first-observed provenance.
        context_not_before = self.calendar.session_close(sessions[-1]) + self.pd.Timedelta(minutes=15)
        if context_not_before.to_pydatetime() > utc_now():
            raise ValueError("last context session has not passed the completion buffer")
        # Matches pinned upstream model/kronos.py:531-532, explicitly supplied
        # so this estimate is never mislabeled as observed dollar turnover.
        context["amount"] = context["volume"] * context[["open", "high", "low", "close"]].mean(axis=1)
        if not self.np.isfinite(context["amount"]).all() or (context["amount"] > FLOAT32_MAX).any():
            raise ValueError("derived amount proxy exceeds the model's finite float32 range")
        params = self.manifest["parameters"]
        random.seed(prepared["seed"])
        self.np.random.seed(prepared["seed"])
        self.torch.manual_seed(prepared["seed"])
        if self.device == "mps":
            self.torch.mps.manual_seed(prepared["seed"])
        started = time.perf_counter()
        forecast = self.predictor.predict(
            df=context[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=self.pd.Series(sessions),
            y_timestamp=self.pd.Series(self.pd.to_datetime([prepared["target_session"]])),
            pred_len=params["pred_len"], T=params["T"], top_k=params["top_k"],
            top_p=params["top_p"], sample_count=params["sample_count"], verbose=False,
        )
        if self.device == "mps":
            self.torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        raw = {key: float(value) for key, value in forecast.iloc[0].items()}
        validation_error = None
        try:
            validate_ohlcv(raw, forecast=True)
            if not self.np.isfinite(raw["amount"]) or raw["amount"] < 0:
                raise ValueError("forecast amount must be finite and nonnegative")
        except ValueError as exc:
            validation_error = str(exc)
        # Nonfinite raw values are represented explicitly as null in valid JSON.
        raw = {key: value if self.np.isfinite(value) else None for key, value in raw.items()}
        completed_at = utc_now()
        return {
            "status": "invalid_forecast" if validation_error else "ok",
            "validation_error": validation_error,
            "ticker": prepared["ticker"], "target_session": prepared["target_session"],
            "context_end": prepared["context_end"], "context_start": prepared["bars"][0]["session"],
            "context_rows": len(prepared["bars"]), "price_basis": prepared["price_basis"],
            "input_hash": prepared["input_hash"], "request_key": prepared["request_key"],
            "seed": prepared["seed"], "manifest_hash": manifest_digest(self.manifest),
            "execution_fingerprint": self.execution_fingerprint,
            "source_revision": self.manifest["source"]["revision"],
            "model_revision": self.manifest["model"]["revision"],
            "tokenizer_revision": self.manifest["tokenizer"]["revision"],
            "amount_convention": self.manifest["amount_convention"],
            "output_kind": self.manifest["output_kind"], "parameters": params,
            "generated_at": completed_at.isoformat(),
            "context_not_before": context_not_before.isoformat(),
            "reconstructed": completed_at >= self.calendar.session_open(prepared["target_session"]).to_pydatetime(),
            "device": self.device, "runtime_versions": self.versions,
            "forecast": raw, "inference_seconds": elapsed,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".runtime" / "kronos")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    worker = DailyWorker(args.runtime.resolve(), args.device)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            if message.get("describe"):
                print(json.dumps({"execution_profile": worker.execution_profile,
                                  "execution_fingerprint": worker.execution_fingerprint,
                                  "manifest": worker.manifest, "load_seconds": worker.load_seconds}), flush=True)
                continue
            requests = message["requests"]
            if not isinstance(requests, list):
                raise ValueError("requests must be a list")
            results = []
            for request in requests:
                try:
                    results.append(worker.predict(request))
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    results.append({"status": "invalid_request", "ticker": request.get("ticker") if isinstance(request, dict) else None, "error": str(exc)})
            print(json.dumps({"results": results, "load_seconds": worker.load_seconds}, allow_nan=False), flush=True)
        except (KeyError, TypeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
