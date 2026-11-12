"""Collect pinned daily forecasts from PriceStore; resumable, offline and optional."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import exchange_calendars as xcals
import pandas as pd

from stock_picker.ingestion.kronos_client import KronosWorkerClient
from stock_picker.log import get_logger
from stock_picker.storage.kronos_store import KronosStore, PRICE_BASIS
from stock_picker.storage.paths import data_root
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from tools.kronos.contract import prepare_request

logger = get_logger(__name__)


def normalized_history(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    index = pd.DatetimeIndex(history.index)
    if index.tz is not None:
        index = index.tz_convert("America/New_York").tz_localize(None)
    history.index = index.normalize()
    if not history.index.is_unique:
        raise ValueError("duplicate stored sessions")
    return history.sort_index()


def request_for_history(ticker: str, target: str, history: pd.DataFrame,
                        manifest: dict, fingerprint: str) -> dict:
    """Use only the 128 prior XNYS bars; target/future prices never enter identity.

    API: https://github.com/gerrymanoim/exchange_calendars (sessions_window includes
    exactly abs(count) sessions). Callers may reuse a normalized history.
    """
    calendar = xcals.get_calendar("XNYS")
    previous = calendar.previous_session(target)
    sessions = calendar.sessions_window(previous, -manifest["parameters"]["lookback"])
    context = history.reindex(sessions)
    columns = ("Open", "High", "Low", "Close", "Volume")
    if context[list(columns)].isna().any().any():
        raise ValueError("missing required consecutive history")
    bars = [{"session": session.date().isoformat(),
             **dict(zip((name.lower() for name in columns), map(float, values)))}
            for session, values in zip(sessions, context[list(columns)].itertuples(index=False, name=None))]
    return prepare_request({"ticker": ticker, "target_session": target,
                            "price_basis": PRICE_BASIS, "bars": bars}, manifest, fingerprint)


def retained_sessions(through: str) -> list[str]:
    """Match the feature snapshot's retained year, excluding the lower boundary."""
    lower = pd.Timestamp(through) - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    return [day.date().isoformat() for day in xcals.get_calendar("XNYS").sessions_in_range(lower, through)]


def latest_collectible_target(now: datetime | None = None) -> str:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    calendar = xcals.get_calendar("XNYS")
    recent = calendar.sessions_in_range(now.date() - pd.Timedelta(days=10), now.date())
    completed = [day for day in recent if calendar.session_close(day) + pd.Timedelta(minutes=15) <= now]
    return calendar.next_session(completed[-1]).date().isoformat()


def collect(tickers: list[str], targets: list[str], prices: PriceStore, store: KronosStore,
            client: KronosWorkerClient, *, batch_size: int = 32, max_seconds: float = 600,
            max_requests: int | None = None, progress=None, expected_fingerprint: str | None = None,
            refresh_inputs: bool = False, refresh_since: str | None = None) -> dict:
    """A session-major pass catches missed sessions and resumes exact input keys.

    Missing context is recorded but never excludes a ticker from downstream models.
    Transport failures stay retryable; one restart is allowed per invocation.
    """
    started = time.monotonic()
    coverage_targets = sorted(set(targets))
    description = client.start()
    try:
        if expected_fingerprint and description["execution_fingerprint"] != expected_fingerprint:
            raise ValueError("worker does not match requested execution fingerprint")
        contract = store.register_profile(description)
        store.set_collection_contract(contract)
        manifest = description["manifest"]
        done = store.completed_keys(contract, coverage_targets[0], coverage_targets[-1]) if targets and not refresh_inputs else set()
        if refresh_since is not None:
            done = {(ticker, target) for ticker, target in done if target < refresh_since}
        counts = {"cached": 0, "ok": 0, "invalid_forecast": 0, "missing_context": 0, "worker_error": 0, "invalid_request": 0}
        histories, sources = {}, {}
        for ticker in tickers:
            try:
                histories[ticker] = normalized_history(prices.read(ticker))
                # Hash the data used by the collector, not a file that ingestion
                # could replace after the read. Exact 128-row input_hash is primary.
                sources[ticker] = {"kind": PRICE_BASIS, "read_at": datetime.now(timezone.utc).isoformat(),
                    "history_hash": hashlib.sha256(pd.util.hash_pandas_object(histories[ticker], index=True).values.tobytes()).hexdigest()}
            except (OSError, ValueError, TypeError):
                histories[ticker] = None
        pending: list[dict] = []
        attempted = 0
        restarted = False
        stopped = False

        def flush() -> None:
            nonlocal restarted, attempted
            if not pending:
                return
            try:
                results = client.predict(pending)
            except (OSError, RuntimeError, ValueError, TimeoutError):
                if restarted:
                    results = [{"status": "worker_error"} for _ in pending]
                else:
                    restarted = True
                    try:
                        replacement = client.start()
                        if store.register_profile(replacement) != contract:
                            raise ValueError("restarted worker changed execution contract")
                        results = client.predict(pending)
                    except (OSError, RuntimeError, ValueError, TimeoutError):
                        results = [{"status": "worker_error"} for _ in pending]
            for request, result in zip(pending, results):
                ticker, target = request["ticker"], request["target_session"]
                if result.get("status") in ("ok", "invalid_forecast"):
                    store.save(contract, request, result, sources[ticker])
                    counts[result["status"]] += 1
                else:
                    status = "invalid_request" if result.get("status") == "invalid_request" else "worker_error"
                    store.record_attempt(contract, ticker, target, status, request["request_key"],
                                         json.dumps(result, allow_nan=False)[:2000])
                    counts[status] += 1
                attempted += 1
            pending.clear()
            if progress:
                progress({**counts, "seconds": round(time.monotonic() - started, 1)})

        for target in targets:
            for ticker in tickers:
                if (ticker, target) in done:
                    counts["cached"] += 1
                    continue
                if time.monotonic() - started >= max_seconds or (max_requests is not None and attempted + len(pending) >= max_requests):
                    stopped = True
                    break
                try:
                    history = histories[ticker]
                    if history is None:
                        raise ValueError("missing price history")
                    request = request_for_history(ticker, target, history, manifest, contract["execution_fingerprint"])
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    store.record_attempt(contract, ticker, target, "missing_context", detail=str(exc)[:300])
                    counts["missing_context"] += 1
                    continue
                cached = store.get(contract, ticker, target, request["request_key"])
                if cached is not None:
                    store.record_attempt(contract, ticker, target, cached["status"], request["request_key"])
                    counts["cached"] += 1
                    continue
                pending.append(request)
                if len(pending) >= batch_size:
                    flush()
            if stopped:
                break
        flush()
        return {"contract": contract, "counts": counts, "stopped_at_limit": stopped,
                "seconds": round(time.monotonic() - started, 2),
                "coverage": store.coverage(contract, tickers, coverage_targets)}
    finally:
        client.close()


def collect_daily(*, max_seconds: float = 600) -> dict:
    """Nightly hook: missing runtime is optional; established history is caught up."""
    root = data_root().parent
    if not (root / ".runtime/kronos/venv/bin/python").is_file():
        return {"status": "unavailable", "reason": "optional runtime absent"}
    store = KronosStore()
    contract = store.collection_contract()
    if contract is None:
        return {"status": "unconfigured", "reason": "run collector once to pin execution contract"}
    description = store.description(contract)
    target = latest_collectible_target()
    lower = max(store.first_target(contract) or target, retained_sessions(target)[0])
    targets = [day.date().isoformat() for day in xcals.get_calendar("XNYS").sessions_in_range(lower, target)]
    # Today's next-open forecast gets the budget first, then recent corrections,
    # then older holes. A large historical backlog cannot starve tomorrow.
    targets = list(reversed(targets[-5:])) + targets[:-5]
    refresh_since = xcals.get_calendar("XNYS").sessions_window(target, -5)[0].date().isoformat()
    return collect(UniverseStore().active_tickers(), targets, PriceStore(), store,
                   KronosWorkerClient(root, description["execution_profile"]["device"], timeout=120),
                   max_seconds=max_seconds, expected_fingerprint=contract["execution_fingerprint"],
                   refresh_since=refresh_since,
                   progress=lambda row: logger.info("Kronos collection %s", row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--daily", action="store_true", help="catch up the configured active universe without training")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--cache-dir", type=Path, default=data_root() / "kronos")
    parser.add_argument("--price-dir", type=Path, default=data_root() / "prices")
    parser.add_argument("--runtime-root", type=Path, default=data_root().parent)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--refresh-inputs", action="store_true", help="recheck completed contexts after source corrections")
    args = parser.parse_args()
    # `bazel run` executes under runfiles. User-relative paths still refer to
    # the repository where the command was invoked, like all normal stores.
    for name in ("cache_dir", "price_dir", "runtime_root", "report"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            setattr(args, name, data_root().parent / value)
    if args.batch_size < 1 or args.max_seconds <= 0 or (args.max_requests is not None and args.max_requests < 1):
        parser.error("batch size and limits must be positive")
    if args.daily:
        if (args.start or args.end or args.tickers or args.max_requests or args.refresh_inputs
                or args.cache_dir != data_root() / "kronos" or args.price_dir != data_root() / "prices"
                or args.runtime_root != data_root().parent):
            parser.error("--daily uses the configured cache/runtime and entire active universe; use a date range for custom collection")
        report = collect_daily(max_seconds=args.max_seconds)
    else:
        if not args.start or not args.end:
            parser.error("provide --start and --end, or --daily")
        targets = [day.date().isoformat() for day in xcals.get_calendar("XNYS").sessions_in_range(args.start, args.end)]
        if not targets:
            parser.error("date range has no XNYS sessions")
        report = collect(args.tickers or UniverseStore().active_tickers(), targets, PriceStore(args.price_dir),
                         KronosStore(args.cache_dir), KronosWorkerClient(args.runtime_root, args.device),
                         batch_size=args.batch_size, max_seconds=args.max_seconds, max_requests=args.max_requests,
                         expected_fingerprint=args.expected_fingerprint,
                         refresh_inputs=args.refresh_inputs,
                         progress=lambda row: print(json.dumps(row), flush=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report.get("counts", {}).get("worker_error") or report.get("counts", {}).get("invalid_request"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
