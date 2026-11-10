"""Freeze score-selected cohorts and raw market data; never evaluate returns.

Uses the existing Massive client with explicit raw-price requests. Provider
contract: https://massive.com/docs/rest/stocks/aggregates/custom-bars
Library contracts: numpy.load(..., allow_pickle=False) and pandas.read_parquet
with an explicit columns selection, per their official API documentation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
EXPERIMENT = REPO / "research/morning_dip_model_2026_11_09"
OLD = REPO / "research/averaging_down_2026_11_06/rank_reconciliation"
PRIOR = REPO / "research/rank_dip_entry_2026_11_09"
ET = ZoneInfo("America/New_York")
THREAD_LOCAL = threading.local()
STOP = threading.Event()
LIMITS = {"minute": 750, "daily": 1700, "metadata": 5}


def repo_imports():
    sys.path.insert(0, str(REPO / "python"))
    runfiles = REPO / "bazel-bin/python/stock_picker/api/main.runfiles"
    sys.path.extend(str(p) for p in sorted(runfiles.glob("*/site-packages")))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=None if path.suffix == ".gz" else 2,
                      allow_nan=False, separators=(",", ":") if path.suffix == ".gz" else None) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(raw)
        temporary = Path(stream.name)
    temporary.replace(path)


def read(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze():
    """Select from archived scores only, before any market outcome analysis."""
    if (ROOT / "cohort.json").exists():
        raise ValueError("Frozen cohort already exists; do not overwrite it")
    import numpy as np
    import pandas as pd
    from stock_picker.ingestion.session import cash_session_window

    protocol = {"frozen_at_utc": now(), "purpose": "Outcome-independent acquisition for dip timing research",
                "cohorts": {"live": "saved Rank1–5", "research_baseline": "outer-period Rank1, 132 raw inputs",
                            "research_all_dips": "outer-period Rank1, 132 existing +21 new dip inputs"},
                "trading_expense_fraction": 0, "new_request_caps": LIMITS, "maximum_concurrency": 4,
                "no_model_training": True, "no_named_ticker_exclusions": True,
                "availability_policy_owner": "analysis protocol; primary early-open assumption, not recorded run logs",
                "recorded_run_warning": "Log timestamps are diagnostic run events, not certified first publication",
                "provider_docs": "https://massive.com/docs/rest/stocks/aggregates/custom-bars"}
    dump(ROOT / "acquisition_protocol.json", protocol)
    calendar_cache = {}
    def calendar(day_string):
        if day_string not in calendar_cache:
            day = date.fromisoformat(day_string)
            bounds = cash_session_window(datetime.combine(day, datetime.min.time(), ET))
            require(bounds is not None, "Non-session in frozen cohort")
            previous = day - timedelta(days=1)
            while cash_session_window(datetime.combine(previous, datetime.min.time(), ET)) is None:
                previous -= timedelta(days=1)
            calendar_cache[day_string] = {"previous_session": str(previous), "open_at": bounds.open_at.isoformat(),
                                          "close_at": bounds.close_at.isoformat()}
        return calendar_cache[day_string]
    prior_rows = {r["session"]: r for r in read(PRIOR / "cohort.json")}
    rows = []
    for path in sorted((REPO / "data/buy_signals").glob("*-rank.json")):
        doc = read(path)
        session = doc["as_of"]
        if session not in prior_rows:
            continue
        require(doc["signals"][0]["ticker"] == prior_rows[session]["ticker"], "Saved Rank1 identity changed")
        scores = [r["predicted_return"] for r in doc["signals"]]
        require(all(math.isfinite(s) for s in scores) and scores == sorted(scores, reverse=True), "Invalid live score order")
        require(len(doc["signals"]) >= 5, "Insufficient live ranks")
        for rank, pick in enumerate(doc["signals"][:5], 1):
            rows.append({"cohort": "live", "session": session, "ticker": pick["ticker"], "rank": rank,
                         "rank_score": float(pick["predicted_return"]), **calendar(session),
                         "recorded_run_at_et": prior_rows[session].get("published_at_et"),
                         "recorded_run_semantics": "diagnostic log event; not certified first publication",
                         "training_end": None, "source_provenance": {"path": str(path.relative_to(REPO)), "sha256": digest(path)}})
    require(len(rows) == 185 and len({r["session"] for r in rows}) == 37, "Unexpected live cohort size")
    manifest_path = EXPERIMENT / "model_run/manifest.json"
    manifest = read(manifest_path)
    run_identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    baseline_path = EXPERIMENT / "baseline.parquet"
    require(digest(baseline_path) == manifest["inputs"]["baseline"], "Frozen baseline changed")
    panel = pd.read_parquet(baseline_path, columns=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    require(not panel.duplicated(["date", "ticker"]).any(), "Duplicate panel identities")
    def identity(frame):
        return hashlib.sha256(pd.util.hash_pandas_object(frame[["date", "ticker"]], index=False).to_numpy("uint64").tobytes()).hexdigest()
    checks = []
    dates = np.sort(panel.date.unique())
    for block in manifest["protocol"]["rank_outer_blocks"]:
        train_dates = dates[dates < np.datetime64(block["from"])][-manifest["protocol"]["training_sessions"]:]
        train = panel.loc[panel.date.isin(train_dates)]
        test = panel.loc[panel.date.between(block["from"], block["through"])]
        for variant, cohort in (("baseline", "research_baseline"), ("all", "research_all_dips")):
            checkpoint = EXPERIMENT / "model_run/checkpoints" / f"rank_{block['name']}_{variant}"
            metadata_path, array_path = checkpoint.with_suffix(".json"), checkpoint.with_suffix(".npy")
            metadata = read(metadata_path)
            scores = np.load(array_path, allow_pickle=False)
            context = metadata["context"]
            require(metadata["run"] == run_identity and digest(array_path) == metadata["sha256"], "Checkpoint identity mismatch")
            require(identity(train) == context["train_rows"] and identity(test) == context["test_rows"], "Checkpoint row alignment mismatch")
            require(context["fit"] is False and train.date.max() < test.date.min(), "Training chronology mismatch")
            require(str(train.date.max()) == context["fit_end"] and str(test.date.min()) == context["score_start"], "Checkpoint date provenance mismatch")
            require(scores.shape == (len(test),) and list(scores.shape) == metadata["shape"] and np.isfinite(scores).all(), "Invalid score array")
            selected = test.assign(score=scores, array_position=np.arange(len(test))).sort_values(
                ["date", "score", "ticker"], ascending=[True, False, True], kind="stable").groupby("date", sort=False).head(1)
            training_end = train.date.max().date().isoformat()
            provenance = {"array_path": str(array_path.relative_to(REPO)), "array_sha256": metadata["sha256"],
                          "metadata_path": str(metadata_path.relative_to(REPO)), "metadata_sha256": digest(metadata_path),
                          "test_rows_sha256": context["test_rows"], "train_rows_sha256": context["train_rows"],
                          "block": block["name"], "raw_feature_count": len(context["columns"])}
            checks.append({"cohort": cohort, "block": block["name"], "training_end": training_end,
                           "test_start": test.date.min().date().isoformat(), "test_end": test.date.max().date().isoformat(),
                           "test_rows": len(test), "selected_dates": len(selected), **provenance})
            for record in selected.itertuples(index=False):
                session = record.date.date().isoformat()
                rows.append({"cohort": cohort, "session": session, "ticker": record.ticker, "rank": 1,
                             "rank_score": float(record.score), **calendar(session), "recorded_run_at_et": None,
                             "recorded_run_semantics": "research model; no actual publication event",
                             "training_end": training_end, "source_provenance": {**provenance, "array_position": int(record.array_position)}})
    rows.sort(key=lambda r: (r["cohort"], r["session"], r["rank"]))
    require(Counter(r["cohort"] for r in rows) == {"live": 185, "research_baseline": 342, "research_all_dips": 342}, "Unexpected cohort counts")
    require(len({(r["cohort"], r["session"], r["rank"]) for r in rows}) == len(rows), "Duplicate cohort keys")
    dump(ROOT / "cohort.json", rows)
    dump(ROOT / "cohort_provenance.json", {"frozen_at_utc": now(), "cohort_sha256": digest(ROOT / "cohort.json"),
         "baseline_sha256": manifest["inputs"]["baseline"], "manifest_path": str(manifest_path.relative_to(REPO)),
         "manifest_sha256": digest(manifest_path), "run_identity": run_identity, "checkpoint_checks": checks,
         "cohort_counts": dict(Counter(r["cohort"] for r in rows)),
         "unique_market_stock_days": len({(r["ticker"], r["session"]) for r in rows}),
         "mrcy_rows": [{k: r[k] for k in ("cohort", "session", "rank")} for r in rows if r["ticker"] == "MRCY"],
         "selection_versions_immutable_live": False, "research_scores_not_live_history": True})
    print(json.dumps({"stage": "frozen", "rows": len(rows), "unique_stock_days": len({(r['ticker'],r['session']) for r in rows})}), flush=True)


def get_client():
    from stock_picker.ingestion.massive_overnight import MassiveOvernightClient
    if not hasattr(THREAD_LOCAL, "client"):
        THREAD_LOCAL.client = MassiveOvernightClient()
    return THREAD_LOCAL.client


def daily_ok(record, ticker, session):
    payload = record.get("payload", {})
    return (record.get("adjusted", False) is False and payload.get("status") == "OK"
            and payload.get("symbol") == record.get("provider_ticker", ticker) and payload.get("from") == session
            and all(isinstance(payload.get(k), (int, float)) and math.isfinite(payload[k]) and payload[k] > 0 for k in ("open", "close")))


def prepare_sources():
    rows = read(ROOT / "cohort.json")
    needed_minute = {(r["ticker"], r["session"]) for r in rows}
    needed_daily = {(r["ticker"], d) for r in rows for d in (r["session"], r["previous_session"])}
    daily = {}
    cache_paths = [OLD / "daily_price_snapshot.json", REPO / "research/rank_score_all_picks_2026_11_09/price_snapshot.json", PRIOR / "daily_prices.json"]
    for path in cache_paths:
        source_hash = digest(path)
        for record in read(path):
            key = (record["ticker"], record["date"])
            if key in needed_daily and daily_ok(record, *key):
                daily[key] = {**record, "adjusted": False, "source_path": str(path.relative_to(REPO)), "source_sha256": source_hash,
                              "adjustment_evidence": "explicit false field or existing raw-request acquisition contract"}
    minute = {}
    archives = sorted((OLD / "minute_bars").glob("*.json.gz")) + [PRIOR / "sources/IONS_2026-11-09_minute.json"]
    for path in archives:
        doc = read(path)
        require(doc.get("adjusted") is False, "Adjusted minute archive is forbidden")
        ticker = doc["ticker"]
        counts = Counter(datetime.fromtimestamp(b["t"] / 1000, ET).date().isoformat() for b in doc.get("bars", []) if isinstance(b, dict) and isinstance(b.get("t"), (int,float)))
        keys = [(ticker, d) for d in counts if (ticker, d) in needed_minute]
        if keys:
            source_hash = digest(path)
            for key in keys:
                minute.setdefault(key, {"ticker": ticker, "session": key[1], "path": str(path.relative_to(REPO)), "sha256": source_hash,
                    "adjusted": False, "complete": True, "completion_evidence": "legacy complete-request archive; original response status pages not retained",
                    "source_entry_date": doc.get("entry_date"), "source_end_date": doc.get("end_date", doc.get("entry_date")),
                    "bar_count_date": counts[key[1]], "cached_existing": True})
    tasks = []
    for kind, keys, cached in (("daily", needed_daily, daily), ("minute", needed_minute, minute)):
        missing = sorted(keys - cached.keys())
        require(len(missing) <= LIMITS[kind], "Acquisition request cap exceeded")
        tasks.extend((kind, ticker, session) for ticker, session in missing)
    dump(ROOT / "daily_prices.json", [daily[k] for k in sorted(daily)])
    dump(ROOT / "minute_manifest.json", [minute[k] for k in sorted(minute)])
    dump(ROOT / "acquisition_plan.json", {"created_at_utc": now(), "cohort_sha256": digest(ROOT / "cohort.json"),
         "needed_daily_keys": len(needed_daily), "needed_minute_keys": len(needed_minute),
         "existing_daily": len(daily), "existing_minute": len(minute), "missing_by_kind": dict(Counter(t[0] for t in tasks)),
         "tasks": [{"kind": k, "ticker": t, "session": s} for k,t,s in tasks]})
    print(json.dumps({"stage": "planned", "missing": dict(Counter(t[0] for t in tasks)), "existing_daily": len(daily), "existing_minute": len(minute)}), flush=True)


def fetch_one(task):
    from stock_picker.ingestion.massive_overnight import API_ROOT
    kind, ticker, session = (task[k] for k in ("kind", "ticker", "session"))
    provider = task.get("provider_ticker", ticker)
    suffix = f"__{provider}" if provider != ticker else ""
    path = ROOT / "sources" / kind / (f"{ticker}_{session}{suffix}.json" + (".gz" if kind == "minute" else ""))
    if path.exists():
        existing = read(path)
        if existing.get("complete"):
            return kind, ticker, session, existing, path
    record = {"ticker": ticker, "session": session, "date": session, "entry_date": session,
              "provider_ticker": provider, "adjusted": False, "retrieved_at": now(), "complete": False, "request_attempts": 0}
    if task.get("symbol_alias_provenance"):
        record["symbol_alias_provenance"] = task["symbol_alias_provenance"]
    if STOP.is_set():
        return kind, ticker, session, {**record, "error_type": "StoppedAfterSystemicProviderError"}, None
    endpoint = f"/v1/open-close/{provider}/{session}" if kind == "daily" else f"/v2/aggs/ticker/{provider}/range/1/minute/{session}/{session}"
    record["endpoint"] = endpoint
    for attempt in range(1, 3):
        record["request_attempts"] = attempt
        try:
            client = get_client()
            if kind == "daily":
                payload = client._get_payload(API_ROOT + endpoint, {"adjusted": "false"})
                record.update(payload=payload, http_status=200, payload_status=payload.get("status"))
                require(daily_ok(record, ticker, session), "Invalid daily response identity/status/prices")
            else:
                bars, pages = [], []
                for page in client._pages(endpoint, {"adjusted": "false", "sort": "asc", "limit": 50000}):
                    require(page.get("status") == "OK" and page.get("ticker", provider) == provider, "Invalid minute response identity/status")
                    require(page.get("adjusted", False) is False, "Minute response unexpectedly adjusted")
                    values = page.get("results") or []
                    require(isinstance(values, list) and all(isinstance(v, dict) for v in values), "Invalid minute results")
                    bars.extend(values)
                    pages.append({"http_status": 200, "payload_status": page.get("status"), "results_count": len(values),
                                  "has_next_page": page.get("next_url") is not None})
                require(pages and not pages[-1]["has_next_page"], "Incomplete minute pagination")
                record.update(bars=bars, page_metadata=pages, bar_count=len(bars))
            record["complete"] = True
            break
        except Exception as error:
            # Existing client sanitizes transport errors; never retain exception strings or keyed URLs.
            text = str(error)
            record["error_type"] = type(error).__name__
            match = re.fullmatch(r"Massive HTTP (\d{3})", text)
            status = int(match.group(1)) if match else None
            if status is not None:
                record["http_status"] = status
            if status in (401,403,429):
                STOP.set()
                break
            if attempt == 1 and (status in (500,502,503,504) or text == "Massive request failed"):
                time.sleep(1)
                continue
            break
    dump(path, record)
    return kind, ticker, session, record, path


def fetch_sources():
    plan = read(ROOT / "acquisition_plan.json")
    daily = {(r["ticker"], r["date"]): r for r in read(ROOT / "daily_prices.json")}
    minute = {(r["ticker"], r["session"]): r for r in read(ROOT / "minute_manifest.json")}
    records = []
    def checkpoint():
        dump(ROOT / "daily_prices.json", [daily[k] for k in sorted(daily)])
        dump(ROOT / "minute_manifest.json", [minute[k] for k in sorted(minute)])
        progress = {"updated_at_utc": now(), "tasks_finished": len(records), "tasks_planned": len(plan["tasks"]),
                    "successful": sum(r["complete"] for r in records), "failed": sum(not r["complete"] for r in records),
                    "stopped_for_provider_error": STOP.is_set(), "daily_keys": len(daily), "minute_keys": len(minute),
                    "records": records}
        dump(ROOT / "acquisition_progress.json", progress)
        print(json.dumps({k:v for k,v in progress.items() if k != "records"}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_one, task) for task in plan["tasks"]]
        for future in as_completed(futures):
            kind, ticker, session, record, path = future.result()
            reference = {"kind": kind, "ticker": ticker, "session": session, "complete": record["complete"],
                         "path": str(path.relative_to(REPO)) if path else None, "error_type": record.get("error_type"),
                         "http_status": record.get("http_status"), "request_attempts": record.get("request_attempts", 0)}
            records.append(reference)
            if path:
                sha = digest(path)
                if kind == "daily":
                    daily[(ticker, session)] = {**record, "source_path": str(path.relative_to(REPO)), "source_sha256": sha}
                else:
                    minute[(ticker, session)] = {"ticker": ticker, "session": session, "path": str(path.relative_to(REPO)),
                        "sha256": sha, "adjusted": False, "complete": record["complete"],
                        "completion_evidence": "validated HTTP/payload status and complete bounded pagination",
                        "source_entry_date": session, "source_end_date": session,
                        "bar_count_date": record.get("bar_count", 0), "cached_existing": False,
                        "error_type": record.get("error_type")}
            if len(records) % 50 == 0:
                checkpoint()
    checkpoint()
    require(not STOP.is_set(), "Collection stopped after systemic provider error")


def fetch_splits():
    """Validate share-unit continuity with bounded explicit date-filter pages.

    As in the repository client's action reader, an opaque next-page cursor is
    not trusted to retain filters. Bisect the date window instead.
    """
    from stock_picker.ingestion.massive_overnight import API_ROOT, SPLITS_PATH
    path = ROOT / "split_metadata.json"
    if path.exists():
        require(read(path).get("complete"), "Existing split metadata is incomplete")
        return
    progress = read(ROOT / "acquisition_progress.json")
    require(progress["tasks_finished"] == progress["tasks_planned"], "Finish the four-worker fetch before metadata requests")
    rows = read(ROOT / "cohort.json")
    start = min(date.fromisoformat(r["previous_session"]) for r in rows)
    end = max(date.fromisoformat(r["session"]) for r in rows)
    tickers = {r["ticker"] for r in rows}
    pending, pages, events = [(start, end)], [], []
    result = {"retrieved_at_utc": now(), "endpoint": SPLITS_PATH, "coverage_start": str(start),
              "coverage_end": str(end), "complete": False, "relevant_events": [], "pages": pages,
              "scope": "splits only; does not certify dividends, spinoffs or ticker continuity"}
    try:
        while pending:
            require(len(pages) < LIMITS["metadata"], "Split metadata request cap exceeded")
            left, right = pending.pop()
            params = {"execution_date.gte": str(left), "execution_date.lte": str(right),
                      "sort": "execution_date.asc", "limit": 5000}
            payload = get_client()._get_payload(API_ROOT + SPLITS_PATH, params)
            values = payload.get("results")
            require(payload.get("status") == "OK" and isinstance(values, list), "Invalid split metadata status/results")
            require(all(isinstance(r, dict) and isinstance(r.get("ticker"), str)
                        and str(left) <= r.get("execution_date", "") <= str(right) for r in values), "Split page filter mismatch")
            following = payload.get("next_url") is not None
            pages.append({"params": params, "http_status": 200, "payload_status": payload["status"],
                          "results_count": len(values), "has_next_page": following,
                          "response_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()})
            if following:
                require(left < right, "One-day split window is paginated")
                middle = left + (right - left) // 2
                pending.extend([(middle + timedelta(days=1), right), (left, middle)])
            else:
                events.extend(r for r in values if r["ticker"] in tickers)
        result.update(complete=True, relevant_events=events)
    except Exception as error:
        result["error_type"] = type(error).__name__
    dump(path, result)
    print(json.dumps({"stage": "split_metadata", "complete": result["complete"], "pages": len(pages), "relevant_events": len(events)}), flush=True)


def repair_symbols():
    """Verify the existing dot/hyphen class-share convention before re-querying."""
    from stock_picker.ingestion.massive_overnight import API_ROOT
    rows = read(ROOT / "cohort.json")
    symbols = sorted({r["ticker"] for r in rows if "-" in r["ticker"]})
    aliases_path = ROOT / "symbol_aliases.json"
    if aliases_path.exists():
        require(all(r.get("acquisition_complete", r["complete"]) for r in read(aliases_path)),
                "Existing alias requests need finalization, not automatic repetition")
        return
    aliases = []
    daily = {(r["ticker"], r["date"]): r for r in read(ROOT / "daily_prices.json")}
    minute = {(r["ticker"], r["session"]): r for r in read(ROOT / "minute_manifest.json")}
    metadata_used = len(read(ROOT / "split_metadata.json")["pages"])
    for ticker in symbols:
        relevant = [r for r in rows if r["ticker"] == ticker]
        provider = ticker.replace("-", ".")
        require(provider.replace(".", "-") == ticker, "Class-share mapping mismatch")
        session = max(r["session"] for r in relevant)
        require(metadata_used < LIMITS["metadata"], "Metadata request cap exceeded")
        endpoint = f"/v3/reference/tickers/{provider}"
        payload = get_client()._get_payload(API_ROOT + endpoint, {"date": session})
        metadata_used += 1
        result = payload.get("results", {})
        require(payload.get("status") == "OK" and result.get("ticker") == provider
                and result.get("market") == "stocks", "Dated provider symbol identity not verified")
        source_path = ROOT / "sources" / "symbols" / f"{ticker}_{session}.json"
        dump(source_path, {"ticker": ticker, "provider_ticker": provider, "date": session,
                          "endpoint": endpoint, "http_status": 200, "payload_status": payload["status"],
                          "retrieved_at": now(), "payload": payload})
        provenance = {"path": str(source_path.relative_to(REPO)), "sha256": digest(source_path),
                      "mapping_contract": "python/stock_picker/ingestion/polygon_client.py:123-129",
                      "dated_identity_verified": True}
        entry = {"ticker": ticker, "provider_ticker": provider, "complete": False,
                 "identity": provenance, "repair_requests": []}
        tasks = [("daily", d) for d in sorted({d for r in relevant for d in (r["session"], r["previous_session"])})]
        tasks += [("minute", d) for d in sorted({r["session"] for r in relevant})]
        for kind, day in tasks:
            _, _, _, record, path = fetch_one({"kind": kind, "ticker": ticker, "session": day,
                                               "provider_ticker": provider, "symbol_alias_provenance": provenance})
            require(path is not None, "Alias request stopped")
            reference = {"kind": kind, "session": day, "complete": record["complete"],
                         "path": str(path.relative_to(REPO)), "sha256": digest(path)}
            entry["repair_requests"].append(reference)
            if kind == "daily":
                daily[(ticker, day)] = {**record, "source_path": reference["path"], "source_sha256": reference["sha256"]}
            else:
                minute[(ticker, day)] = {"ticker": ticker, "provider_ticker": provider, "session": day,
                    "path": reference["path"], "sha256": reference["sha256"], "adjusted": False,
                    "complete": record["complete"], "source_entry_date": day, "source_end_date": day,
                    "completion_evidence": "validated provider class-share identity and complete raw request",
                    "symbol_alias_provenance": provenance, "bar_count_date": record.get("bar_count", 0), "cached_existing": False}
        entry["complete"] = all(r["complete"] for r in entry["repair_requests"])
        aliases.append(entry)
    dump(ROOT / "daily_prices.json", [daily[k] for k in sorted(daily)])
    dump(ROOT / "minute_manifest.json", [minute[k] for k in sorted(minute)])
    dump(aliases_path, aliases)
    print(json.dumps({"stage": "symbol_repairs", "aliases": len(aliases), "all_complete": all(r["complete"] for r in aliases)}), flush=True)


def finalize_sources():
    """Audit raw source coverage and hashes without calculating returns.

    Acquisition completion means every planned request has an explicit result;
    it does not turn missing prices into available prices.
    """
    from stock_picker.ingestion.massive_overnight import API_ROOT, SPLITS_PATH
    rows = read(ROOT / "cohort.json")
    aliases = read(ROOT / "symbol_aliases.json")
    splits = read(ROOT / "split_metadata.json")
    daily = read(ROOT / "daily_prices.json")
    minute = read(ROOT / "minute_manifest.json")
    progress = read(ROOT / "acquisition_progress.json")
    plan = read(ROOT / "acquisition_plan.json")
    require(progress["tasks_finished"] == progress["tasks_planned"] and not progress["stopped_for_provider_error"],
            "Acquisition has unfinished planned requests")
    aliases_by_ticker = {r["ticker"]: r for r in aliases}
    # The original split response was hashed but its full body was not retained.
    # A narrow alias request closes that source-coverage gap; retain this body.
    alias_pages = []
    for alias in aliases:
        ticker, provider = alias["ticker"], alias["provider_ticker"]
        identity = alias["identity"]
        require(identity["dated_identity_verified"] and digest(REPO / identity["path"]) == identity["sha256"],
                "Provider alias identity changed")
        path = ROOT / "sources" / "symbols" / f"{ticker}_splits.json"
        if path.exists():
            source = read(path)
        else:
            require(len(splits["pages"]) + len(aliases) + len(alias_pages) < LIMITS["metadata"],
                    "Metadata request cap exceeded")
            params = {"ticker": provider, "execution_date.gte": splits["coverage_start"],
                      "execution_date.lte": splits["coverage_end"], "sort": "execution_date.asc", "limit": 5000}
            payload = get_client()._get_payload(API_ROOT + SPLITS_PATH, params)
            source = {"ticker": ticker, "provider_ticker": provider, "endpoint": SPLITS_PATH,
                      "params": params, "http_status": 200, "payload_status": payload.get("status"),
                      "retrieved_at": now(), "payload": payload}
            dump(path, source)
        payload = source["payload"]
        require(payload.get("status") == "OK" and isinstance(payload.get("results"), list)
                and payload.get("next_url") is None, "Incomplete alias split coverage")
        require(all(r.get("ticker") == provider
                    and splits["coverage_start"] <= r.get("execution_date", "") <= splits["coverage_end"]
                    for r in payload["results"]), "Alias split result identity/filter mismatch")
        alias_pages.append({"ticker": ticker, "provider_ticker": provider, "path": str(path.relative_to(REPO)),
                            "sha256": digest(path), "http_status": 200, "payload_status": payload["status"],
                            "has_next_page": False, "results_count": len(payload["results"]),
                            "symbol_alias_provenance": identity})
        existing_events = {(r["ticker"], r["execution_date"]) for r in splits["relevant_events"]}
        for event in payload["results"]:
            if (ticker, event["execution_date"]) not in existing_events:
                splits["relevant_events"].append({**event, "ticker": ticker, "provider_ticker": provider,
                                                  "provider_alias_verified": True,
                                                  "symbol_alias_provenance": identity})
        alias["acquisition_complete"] = True
        alias["all_prices_available"] = alias["complete"]
        alias["provider_alias_verified"] = True
    splits["alias_pages"] = alias_pages
    splits["alias_mapping_complete"] = True
    dump(ROOT / "split_metadata.json", splits)
    dump(ROOT / "symbol_aliases.json", aliases)
    for record in daily + minute:
        if record.get("provider_ticker", record["ticker"]) != record["ticker"]:
            alias = aliases_by_ticker[record["ticker"]]
            require(record["provider_ticker"] == alias["provider_ticker"]
                    and record["symbol_alias_provenance"] == alias["identity"], "Unverified source alias")
            record["provider_alias_verified"] = True
    dump(ROOT / "daily_prices.json", daily)
    dump(ROOT / "minute_manifest.json", minute)
    needed_daily = {(r["ticker"], d) for r in rows for d in (r["session"], r["previous_session"])}
    needed_minute = {(r["ticker"], r["session"]) for r in rows}
    require(len(daily) == len(needed_daily) and {(r["ticker"], r["date"]) for r in daily} == needed_daily,
            "Daily source keys do not match frozen cohort")
    require(len(minute) == len(needed_minute) and {(r["ticker"], r["session"]) for r in minute} == needed_minute,
            "Minute source keys do not match frozen cohort")
    hashes = {}
    def verify(path_string, expected):
        if path_string not in hashes:
            hashes[path_string] = digest(REPO / path_string)
        require(hashes[path_string] == expected, "Source hash mismatch")
    unavailable_daily = []
    for record in daily:
        verify(record["source_path"], record["source_sha256"])
        require(record.get("adjusted") is False, "Daily source has adjusted prices")
        if not daily_ok(record, record["ticker"], record["date"]):
            unavailable_daily.append({k: record.get(k) for k in
                                      ("ticker", "provider_ticker", "date", "error_type", "http_status", "source_path")})
    source_docs, source_coverage = {}, []
    for record in minute:
        verify(record["path"], record["sha256"])
        doc = source_docs.setdefault(record["path"], read(REPO / record["path"]))
        require(record.get("adjusted") is False and doc.get("adjusted") is False, "Minute source has adjusted prices")
        require(doc.get("ticker") == record["ticker"], "Minute source cohort identity mismatch")
        timestamps, regular, invalid = [], 0, 0
        for bar in doc.get("bars", []):
            t = bar.get("t") if isinstance(bar, dict) else None
            if not isinstance(t, (int, float)) or not math.isfinite(t):
                invalid += 1
                continue
            instant = datetime.fromtimestamp(t / 1000, ET)
            if instant.date().isoformat() != record["session"]:
                continue
            timestamps.append(t)
            valid = all(isinstance(bar.get(k), (int, float)) and math.isfinite(bar[k]) and bar[k] > 0 for k in ("o", "h", "l", "c"))
            valid = valid and bar["l"] <= min(bar["o"], bar["c"]) <= max(bar["o"], bar["c"]) <= bar["h"]
            invalid += not valid
            regular += 570 <= instant.hour * 60 + instant.minute < 960
        source_coverage.append({"ticker": record["ticker"], "session": record["session"],
                                "provider_ticker": record.get("provider_ticker", record["ticker"]),
                                "complete": record["complete"], "date_bar_count": len(timestamps),
                                "regular_clock_bar_count": regular, "invalid_bars": invalid,
                                "duplicate_timestamps": len(timestamps) - len(set(timestamps)),
                                "sorted_timestamps": timestamps == sorted(timestamps),
                                "path": record["path"], "sha256": record["sha256"]})
    corpus_files = sorted(p for p in (ROOT / "sources").rglob("*") if p.is_file())
    # Fail without revealing any matched text if a credential-bearing URL slipped in.
    forbidden = re.compile(r"https?://[^\s\"<>]*(?:apiKey|api_key|access_token)=", re.IGNORECASE)
    require(all(not forbidden.search(json.dumps(read(p))) for p in corpus_files), "Credential-bearing source URL found")
    counts = {kind: sum(r["request_attempts"] for r in progress["records"] if r["kind"] == kind)
              for kind in ("daily", "minute")}
    for alias in aliases:
        for request in alias["repair_requests"]:
            counts[request["kind"]] += read(REPO / request["path"])["request_attempts"]
    counts["metadata"] = len(splits["pages"]) + len(aliases) + len(alias_pages)
    require(all(counts[k] <= LIMITS[k] for k in counts), "Final acquisition exceeds request caps")
    provenance = read(ROOT / "cohort_provenance.json")
    require(digest(ROOT / "cohort.json") == provenance["cohort_sha256"], "Frozen cohort changed")
    summary = {"status": "completed", "complete": True, "finalized_at_utc": now(),
               "completion_semantics": "All acquisition tasks have explicit results; absent prices remain unavailable",
               "cohort_sha256": provenance["cohort_sha256"], "cohort_rows": len(rows),
               "cohorts": {c: {"rows": sum(r["cohort"] == c for r in rows),
                                "sessions": len({r["session"] for r in rows if r["cohort"] == c}),
                                "tickers": len({r["ticker"] for r in rows if r["cohort"] == c})}
                           for c in sorted({r["cohort"] for r in rows})},
               "unique_market_stock_days": len(needed_minute), "daily_keys": len(daily),
               "daily_available": len(daily) - len(unavailable_daily), "daily_unavailable": unavailable_daily,
               "minute_keys": len(minute), "minute_sources_complete": sum(r["complete"] for r in minute),
               "minute_nonempty_sessions": sum(r["date_bar_count"] > 0 for r in source_coverage),
               "minute_empty_sessions": [{k: r[k] for k in ("ticker", "session", "path")} for r in source_coverage if not r["date_bar_count"]],
               "minute_invalid_bars": sum(r["invalid_bars"] for r in source_coverage),
               "minute_duplicate_timestamps": sum(r["duplicate_timestamps"] for r in source_coverage),
               "all_source_hashes_verified": True, "all_prices_raw_unadjusted": True,
               "request_attempts_by_kind": counts, "request_caps": LIMITS, "maximum_concurrency": 4,
               "split_metadata_complete": splits["complete"], "alias_split_mapping_complete": True,
               "sources_file_count": len(corpus_files), "sources_bytes": sum(p.stat().st_size for p in corpus_files),
               "credential_url_scan_passed": True, "trading_expense_fraction": 0,
               "manifest_hashes": {name: digest(ROOT / name) for name in
                                   ("cohort.json", "daily_prices.json", "minute_manifest.json", "split_metadata.json", "symbol_aliases.json")},
               "source_coverage": source_coverage,
               "limitations": ["Legacy minute archive HTTP/page statuses were not retained; provenance explicitly labels this.",
                               "Original failed daily sources did not preserve HTTP codes; those codes remain unknown.",
                               "Split metadata does not certify dividends, spinoffs or ticker continuity.",
                               "Regular clock bar counts are diagnostics; analysis uses each session's actual calendar close."]}
    dump(ROOT / "acquisition_audit.json", summary)
    progress.update(status="completed", complete=True, finalized_at_utc=summary["finalized_at_utc"],
                    final_audit="research/rank_dip_timing_2026_11_09/acquisition_audit.json",
                    final_audit_sha256=digest(ROOT / "acquisition_audit.json"))
    dump(ROOT / "acquisition_progress.json", progress)
    print(json.dumps({k: summary[k] for k in ("status", "cohort_rows", "unique_market_stock_days", "daily_keys",
                                              "daily_available", "minute_keys", "minute_nonempty_sessions", "request_attempts_by_kind")}), flush=True)


if __name__ == "__main__":
    repo_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare", "fetch", "splits", "repair-symbols", "finalize"))
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare_sources, "fetch": fetch_sources, "splits": fetch_splits,
     "repair-symbols": repair_symbols, "finalize": finalize_sources}[args.command]()
