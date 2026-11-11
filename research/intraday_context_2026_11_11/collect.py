"""Resumable bounded minute-range archive, independent of browser refresh.

Run with PYTHONPATH=python. Every successful range is immutable, hashed, and
timestamped. Failed ranges are visible and retried on the next invocation.
This is historical collection; a live first-seen collector remains separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, TICKER_PATTERN

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan_ranges(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Merge overlapping 45-calendar-day lookbacks, then bound requests to 28 days."""
    windows = {}
    for row in frame[["ticker", "session"]].drop_duplicates().itertuples(index=False):
        day = date.fromisoformat(row.session)
        windows.setdefault(row.ticker, []).append((day - timedelta(days=45), day))
    windows["SPY"] = [(date.fromisoformat(frame.session.min()) - timedelta(days=45),
                       date.fromisoformat(frame.session.max()))]
    result = []
    for ticker, spans in sorted(windows.items()):
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1] + timedelta(days=1):
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            while start <= end:
                stop = min(end, start + timedelta(days=27))
                result.append((ticker, start.isoformat(), stop.isoformat()))
                start = stop + timedelta(days=1)
    return result


def read_archive(pointer: Path) -> dict:
    metadata = json.loads(pointer.read_text())
    raw = (pointer.parent / metadata["archive"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("archive checksum mismatch")
    document = json.loads(gzip.decompress(raw))
    if not document["complete"] or document["adjusted"] is not False:
        raise ValueError("unverified archive")
    return document


def collect_range(spec: tuple[str, str, str], cache: Path) -> dict:
    ticker, start, end = spec
    if not TICKER_PATTERN.fullmatch(ticker) or date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("invalid requested identity")
    if date.fromisoformat(end) >= datetime.now(ET).date():
        raise ValueError("historical collector requires completed prior dates")
    pointer = cache / f"{ticker}_{start}_{end}.json"
    if pointer.exists():
        document = read_archive(pointer)
        if (document["ticker"], document["start"], document["end"]) != spec:
            raise ValueError("archive identity mismatch")
        return {"ticker": ticker, "start": start, "end": end, "pointer": str(pointer.relative_to(ROOT)), "cached": True}
    path = f"/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
    for attempt in range(3):
        try:
            client = MassiveOvernightClient()
            pages = list(client._pages(path, {"adjusted": "false", "sort": "asc", "limit": 50000}))
            bars = []
            for page in pages:
                if page.get("status") != "OK" or page.get("ticker") != ticker or page.get("adjusted") is not False:
                    raise ValueError("provider status, identity, or price basis mismatch")
                for bar in page.get("results", []):
                    stamp = datetime.fromtimestamp(bar["t"] / 1000, ET)
                    if not start <= stamp.date().isoformat() <= end:
                        raise ValueError("bar outside requested range")
                    if not all(isinstance(bar.get(k), (int, float)) and math.isfinite(bar[k]) for k in ("t", "o", "h", "l", "c", "v")):
                        raise ValueError("invalid bar")
                    if min(bar[k] for k in ("o", "h", "l", "c")) <= 0 or bar["v"] < 0:
                        raise ValueError("nonpositive price or negative volume")
                    bars.append(bar)
            stamps = [bar["t"] for bar in bars]
            if stamps != sorted(set(stamps)):
                raise ValueError("unordered or duplicate minute bars")
            document = {"provider": "massive", "ticker": ticker, "start": start, "end": end,
                        "adjusted": False, "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "complete": True, "completion_evidence": "Validated status, ticker, raw basis, bounded pagination and bar range; empty intervals can be legitimate",
                        "page_count": len(pages), "bars": bars}
            raw = gzip.compress(json.dumps(document, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
            digest = hashlib.sha256(raw).hexdigest()
            archive = f"{ticker}_{start}_{end}_{digest[:16]}.json.gz"
            cache.mkdir(parents=True, exist_ok=True)
            temporary = cache / (archive + "." + uuid4().hex + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(cache / archive)
            atomic_json(pointer, {"archive": archive, "sha256": digest})
            return {"ticker": ticker, "start": start, "end": end, "pointer": str(pointer.relative_to(ROOT)),
                    "cached": False, "bars": len(bars)}
        except Exception as exc:
            if attempt == 2:
                # Never persist request URLs or credentials from exception text.
                return {"ticker": ticker, "start": start, "end": end, "error_type": type(exc).__name__}
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=ROOT / "base.parquet")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    specs = plan_ranges(pd.read_parquet(args.frame))
    atomic_json(ROOT / "collection_plan.json", {"ranges": specs, "workers": args.workers})
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        pending = [pool.submit(collect_range, spec, ROOT / "sources") for spec in specs]
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError("cached source integrity failure; refusing silent replacement") from exc
            if len(results) % 40 == 0 or len(results) == len(specs):
                atomic_json(ROOT / "collection_manifest.json", sorted(results, key=lambda r: (r["ticker"], r["start"])))
                print(json.dumps({"completed": len(results), "total": len(specs), "failures": sum("error_type" in r for r in results)}), flush=True)


if __name__ == "__main__":
    main()
