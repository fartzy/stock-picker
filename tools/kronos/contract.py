"""Pure request/provenance contract shared by worker tests and its callers."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
PRICE_COLUMNS = ("open", "high", "low", "close")
# Upstream converts raw OHLCVA to float32 before normalization.
FLOAT32_MAX = 3.4028234663852886e38


def digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def manifest_digest(manifest: dict) -> str:
    return digest(manifest)


def _session(value: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("sessions must be YYYY-MM-DD")
    return value


def validate_ohlcv(bar: dict, *, forecast: bool = False) -> dict[str, float]:
    result = {}
    for name in (*PRICE_COLUMNS, "volume"):
        value = bar[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite numeric data")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite numeric data") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric data")
        if value > FLOAT32_MAX:
            raise ValueError(f"{name} exceeds the model's float32 range")
        if value < 0 or (name in PRICE_COLUMNS and value == 0):
            raise ValueError(f"{name} must be positive (volume may be zero)")
        result[name] = value
    if result["high"] < max(result["open"], result["close"], result["low"]) or result["low"] > min(result["open"], result["close"]):
        raise ValueError("incoherent forecast OHLC" if forecast else "incoherent historical OHLC")
    return result


def prepare_request(request: dict, manifest: dict, execution_fingerprint: str) -> dict:
    """Reject bad windows, normalize inputs, and derive a stable per-series seed.

    Exchange-session and completed-close checks are additionally enforced by
    the worker. No output, target-day prices, or actual outcome enters this hash.
    """
    ticker = request["ticker"]
    if not isinstance(ticker, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.^=-]{0,15}", ticker):
        raise ValueError("invalid ticker")
    target = _session(request["target_session"])
    if target < manifest["pretraining"]["first_allowed_target"]:
        raise ValueError("target overlaps the documented pretraining period")
    price_basis = request["price_basis"]
    if not isinstance(price_basis, str) or not price_basis.strip():
        raise ValueError("price_basis must explicitly describe the input series")
    bars = request["bars"]
    if len(bars) != manifest["parameters"]["lookback"]:
        raise ValueError("request must contain exactly the configured lookback")
    normalized = []
    prior = None
    for bar in bars:
        session = _session(bar["session"])
        if session >= target or (prior is not None and session <= prior):
            raise ValueError("context must be unique, ascending, and strictly before target")
        normalized.append({"session": session, **validate_ohlcv(bar)})
        prior = session
    payload = {"ticker": ticker, "target_session": target, "price_basis": price_basis, "bars": normalized}
    input_hash = digest(payload)
    identity = {"input_hash": input_hash, "manifest_hash": manifest_digest(manifest)}
    key = digest({**identity, "execution_fingerprint": execution_fingerprint})
    return {
        **payload,
        "context_end": prior,
        "input_hash": input_hash,
        "request_key": key,
        "seed": int(digest(identity)[:8], 16),
    }


def verify_assets(runtime: Path, manifest: dict) -> None:
    for kind in ("source", "model", "tokenizer"):
        for name, expected in manifest[kind]["files"].items():
            path = runtime / kind / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"missing or changed pinned runtime asset: {kind}/{name}")
