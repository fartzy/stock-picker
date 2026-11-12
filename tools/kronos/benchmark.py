"""Benchmark the fixed daily worker against existing local PriceStore histories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import statistics
import sys
import time

from tools.kronos.contract import read_manifest
from tools.kronos.worker import DailyWorker

ROOT = Path(__file__).resolve().parents[2]


def local_requests(target_session: str, limit: int, data_root: Path) -> tuple[list[dict], list[dict], dict]:
    """Use the exact stored values; never imply a verified adjustment convention."""
    sys.path.insert(0, str(ROOT / "python"))
    from stock_picker.storage.universe_store import UniverseStore
    import exchange_calendars as xcals
    import pandas as pd

    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(target_session):
        raise ValueError("target must be an XNYS session")
    previous = calendar.previous_session(target_session)
    context_dates = calendar.sessions_window(previous, -read_manifest()["parameters"]["lookback"])
    active = UniverseStore(data_dir=data_root / "universe").active_tickers()
    priority = [ticker for ticker in ("HURN", "CSQR", "TTMI", "MRCY") if ticker in active]
    ordered = priority + sorted(set(active).difference(priority))
    requests, rejected, source_hashes = [], [], {}
    for ticker in ordered:
        if len(requests) >= limit:
            break
        path = data_root / "prices" / f"{ticker}.parquet"
        try:
            # Parse and hash the same immutable bytes, even if ingestion replaces
            # the PriceStore file while a benchmark is running.
            source_bytes = path.read_bytes()
            history = pd.read_parquet(BytesIO(source_bytes))
            timestamps = pd.DatetimeIndex(history.index)
            if timestamps.tz is not None:
                timestamps = timestamps.tz_convert("America/New_York").tz_localize(None)
            history.index = timestamps.normalize()
            if not history.index.is_unique:
                raise ValueError("duplicate stored sessions")
            context = history.reindex(context_dates)
            if context[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
                raise ValueError("missing required consecutive history")
            bars = [{"session": session.date().isoformat(), **{name.lower(): float(row[name]) for name in ("Open", "High", "Low", "Close", "Volume")}}
                    for session, row in context.iterrows()]
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            rejected.append({"ticker": ticker, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
            continue
        requests.append({"ticker": ticker, "target_session": target_session,
                         "price_basis": "price_store_ohlcv_as_stored", "bars": bars})
        source_hashes[ticker] = hashlib.sha256(source_bytes).hexdigest()
    return requests, rejected, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-session", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".runtime" / "kronos")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        raise ValueError("benchmark limit must be between 1 and 100")
    started = time.perf_counter()
    requests, rejected, source_hashes = local_requests(args.target_session, args.limit, args.data_root)
    if not requests:
        raise ValueError("no stocks have the required local history")
    worker = DailyWorker(args.runtime, args.device)
    results = []
    for request in requests:
        try:
            result = worker.predict(request)
        except (ValueError, KeyError, TypeError) as exc:
            result = {"status": "invalid_request", "ticker": request["ticker"], "error": str(exc)}
        results.append(result)
        print(json.dumps({"ticker": result["ticker"], "status": result["status"], "seconds": result.get("inference_seconds")}), flush=True)
    repeated = []
    # Repeat in reverse order: preceding requests must not change a series' RNG.
    for request, original in list(zip(requests, results))[:3][::-1]:
        if "forecast" not in original:
            continue
        repeat = worker.predict(request)
        repeated.append({"ticker": request["ticker"], "same_forecast": original["forecast"] == repeat["forecast"],
                         "same_request_key": original["request_key"] == repeat["request_key"],
                         "seconds": repeat["inference_seconds"]})
    times = [row["inference_seconds"] for row in results if "inference_seconds" in row]
    warm = times[1:] or times
    mean = statistics.mean(warm) if warm else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "target_session": args.target_session,
        "purpose": "Runtime feasibility only; forecasts generated after the open are reconstructed, not morning observations.",
        "manifest": worker.manifest, "execution_profile": worker.execution_profile,
        "execution_fingerprint": worker.execution_fingerprint,
        "platform": platform.platform(), "requested": args.limit, "eligible": len(requests),
        "history_rejections": rejected, "source_parquet_sha256": source_hashes,
        "load_seconds": worker.load_seconds, "first_forecast_seconds": times[0] if times else None,
        "warm_mean_seconds": mean, "warm_median_seconds": statistics.median(warm) if warm else None,
        "total_seconds": time.perf_counter() - started,
        "sequential_estimate_2000_seconds": mean * 2000 if mean is not None else None,
        "sequential_estimate_500000_hours": mean * 500000 / 3600 if mean is not None else None,
        "estimate_limit": "Linear inference-only extrapolation, not a measured full-universe or backfill run; excludes collection, storage and scheduling overhead.",
        "valid_forecasts": sum(row["status"] == "ok" for row in results),
        "repeat_checks": repeated, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("eligible", "valid_forecasts", "load_seconds", "warm_mean_seconds", "sequential_estimate_2000_seconds", "sequential_estimate_500000_hours")}))
    if not times:
        raise RuntimeError("no request reached inference; inspect the saved validation errors")


if __name__ == "__main__":
    main()
