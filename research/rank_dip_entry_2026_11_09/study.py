"""Fixed-protocol descriptive research; never changes production files or models.

Provider contract: https://massive.com/docs/rest/stocks/aggregates/custom-bars
Minute timestamps mark interval starts; missing intervals are not fabricated.
Uses the repo's existing calendar and raw-price client, not new dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OLD = REPO / "research/averaging_down_2026_11_06/rank_reconciliation"
ET = ZoneInfo("America/New_York")
MINUTE_MS = 60_000
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
ANALYSIS_AMENDMENT = json.loads((ROOT / "analysis_amendment.json").read_text())
if ANALYSIS_AMENDMENT["active_trading_cost_fraction"] != 0:
    raise ValueError("This user-directed study reports price returns without modeled deductions")


def repo_imports():
    """Use the already-built API's pinned dependencies with its Python 3.12."""
    sys.path.insert(0, str(REPO / "python"))
    runfiles = REPO / "bazel-bin/python/stock_picker/api/main.runfiles"
    sys.path.extend(str(p) for p in sorted(runfiles.glob("*/site-packages")))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_cohort():
    from stock_picker.ingestion.session import cash_session_window
    publications = {r["date"]: r["published_at_et"] for r in json.loads((OLD / "publication_times.json").read_text())}
    publications["2026-11-09"] = "2026-11-09T10:07:27.083000-05:00"
    rows = []
    for path in sorted((REPO / "data/buy_signals").glob("*-rank.json")):
        doc = json.loads(path.read_text())
        session = doc["as_of"]
        if not PROTOCOL["cohort"]["first"] <= session <= PROTOCOL["cohort"]["last"]:
            continue
        pick = doc["signals"][0]
        day = date.fromisoformat(session)
        bounds = cash_session_window(datetime.combine(day, datetime.min.time(), ET))
        prior = day - timedelta(days=1)
        while cash_session_window(datetime.combine(prior, datetime.min.time(), ET)) is None:
            prior -= timedelta(days=1)
        score = pick["predicted_return"]
        if not math.isfinite(score):
            raise ValueError("Nonfinite Rank score")
        rows.append({"session": session, "ticker": pick["ticker"], "rank": 1,
                     "rank_score": score, "source": str(path.relative_to(REPO)),
                     "source_sha256": digest(path), "previous_session": prior.isoformat(),
                     "open_at": bounds.open_at.isoformat(), "close_at": bounds.close_at.isoformat(),
                     "published_at_et": publications.get(session),
                     "saved_prev_close": pick.get("prev_close")})
    assert len(rows) == PROTOCOL["cohort"]["expected_sessions"]
    return rows


def fetch_sources():
    from stock_picker.ingestion.massive_overnight import API_ROOT, MassiveOvernightClient
    rows = load_cohort()
    dump(ROOT / "cohort.json", rows)
    daily = {(r["ticker"], r["date"]): {**r, "source": str((OLD / "daily_price_snapshot.json").relative_to(REPO)),
              "source_sha256": digest(OLD / "daily_price_snapshot.json"), "adjusted": False}
             for r in json.loads((OLD / "daily_price_snapshot.json").read_text())}
    needed = {(r["ticker"], d) for r in rows for d in (r["session"], r["previous_session"])}
    missing = sorted(needed - daily.keys())
    assert len(missing) <= PROTOCOL["remote_budget"]["new_daily_requests_max"], len(missing)
    def fetch_daily(key):
        ticker, day = key
        path = ROOT / "sources" / f"daily_{ticker}_{day}.json"
        if path.exists():
            return json.loads(path.read_text())
        endpoint = f"/v1/open-close/{ticker}/{day}"
        result = {"ticker": ticker, "date": day, "adjusted": False, "endpoint": endpoint,
                  "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            result["payload"] = MassiveOvernightClient()._get_payload(API_ROOT + endpoint, {"adjusted": "false"})
        except Exception as error:
            result["error_type"] = type(error).__name__  # Never serialize credential-bearing errors.
        dump(path, result)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(fetch_daily, missing):
            daily[(record["ticker"], record["date"])] = record
    selected_daily = [daily[key] for key in sorted(needed)]
    dump(ROOT / "daily_prices.json", selected_daily)
    manifest = []
    for row in rows:
        files = list((OLD / "minute_bars").glob(f"{row['ticker']}_{row['session']}_*.json.gz"))
        if files:
            path = sorted(files)[0]
            with gzip.open(path, "rt") as handle:
                doc = json.load(handle)
        else:
            assert row["session"] == "2026-11-09" and row["ticker"] == "IONS"
            path = ROOT / "sources/IONS_2026-11-09_minute.json"
            if path.exists():
                doc = json.loads(path.read_text())
            else:
                doc = {"ticker": "IONS", "entry_date": "2026-11-09", "adjusted": False,
                       "retrieved_at": datetime.now(timezone.utc).isoformat()}
                try:
                    doc["bars"] = MassiveOvernightClient().fetch_minute_bars("IONS", date(2026, 11, 9))
                except Exception as error:
                    doc.update(bars=[], error_type=type(error).__name__)
                dump(path, doc)
        assert doc.get("adjusted") is False
        manifest.append({"ticker": row["ticker"], "session": row["session"],
                         "path": str(path.relative_to(REPO)), "sha256": digest(path),
                         "adjusted": False, "retrieved_at": doc.get("retrieved_at")})
    dump(ROOT / "minute_manifest.json", manifest)
    dump(ROOT / "source_manifest.json", {"protocol_sha256": digest(ROOT / "protocol.json"),
         "cohort_sha256": digest(ROOT / "cohort.json"), "needed_daily_keys": len(needed),
         "daily_cache_hits": len(needed) - len(missing), "daily_fetch_keys": len(missing),
         "daily_errors": sum("error_type" in r for r in selected_daily),
         "publication_source": str((OLD / "publication_times.json").relative_to(REPO)),
         "publication_source_sha256": digest(OLD / "publication_times.json"),
         "nov9_publication_source": "/private/tmp/stock-picker-morning-2026-11-09.log:202",
         "nov9_publication_log_excerpt": "2026-11-09 09:07:27,083 INFO stock_picker.training.morning: rank_top=20 published first",
         "nov9_log_timezone": "America/Chicago", "selection_versions_immutable": False})
    print(json.dumps({"saved_sessions": len(rows), "daily_fetch_keys": len(missing),
                      "daily_errors": sum("error_type" in r for r in selected_daily)}))


def fetch_split_metadata():
    """One bounded, explicitly filtered first page; never trust opaque action cursors."""
    from stock_picker.ingestion.massive_overnight import API_ROOT, SPLITS_PATH, MassiveOvernightClient
    rows = json.loads((ROOT / "cohort.json").read_text())
    clarification = ROOT / "protocol_clarification.json"
    if not clarification.exists():
        dump(clarification, {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
             "change": "Validate raw prior-close share units with one bounded date-filtered splits query before outcomes; mark split-crossing references unavailable instead of inventing an adjustment.",
             "threshold_cohort_and_entry_rules_unchanged": True, "additional_metadata_request_limit": 1,
             "existing_protocol_sha256": digest(ROOT / "protocol.json")})
    path = ROOT / "sources/split_metadata.json"
    if path.exists():
        return
    start, end = min(r["previous_session"] for r in rows), max(r["session"] for r in rows)
    params = {"execution_date.gte": start, "execution_date.lte": end,
              "sort": "execution_date.asc", "limit": 5000}
    result = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "endpoint": SPLITS_PATH,
              "params": params, "complete": False, "relevant_events": []}
    try:
        payload = MassiveOvernightClient()._get_payload(API_ROOT + SPLITS_PATH, params)
        events = payload.get("results")
        valid = payload.get("status") == "OK" and isinstance(events, list) and payload.get("next_url") is None
        valid = valid and all(isinstance(r, dict) and start <= str(r.get("execution_date", "")) <= end and isinstance(r.get("ticker"), str) for r in events)
        result.update(complete=bool(valid), response_status=payload.get("status"),
                      raw_response_count=len(events) if isinstance(events, list) else None,
                      raw_response_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                      has_next_page=payload.get("next_url") is not None)
        if valid:
            tickers = {r["ticker"] for r in rows}
            result["relevant_events"] = [r for r in events if r["ticker"] in tickers]
    except Exception as error:
        result["error_type"] = type(error).__name__
    dump(path, result)
    print(json.dumps({"split_metadata_complete": result["complete"], "relevant_events": len(result["relevant_events"])}))


def evaluate(bars, anchor, opened, closed, publication=None, before_noon=True):
    """Signal close is observable before next-minute entry; no optimistic same-bar fill."""
    open_ms, close_ms = (int(t.timestamp() * 1000) for t in (opened, closed))
    valid = {}
    invalid = duplicate = 0
    for bar in bars:
        stamp = bar.get("t")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp % MINUTE_MS:
            invalid += 1
            continue
        if not open_ms <= stamp < close_ms:
            continue
        prices = [bar.get(k) for k in ("o", "h", "l", "c")]
        if not all(positive(v) for v in prices) or not prices[2] <= min(prices[0], prices[3]) <= max(prices[0], prices[3]) <= prices[1]:
            invalid += 1
            continue
        if int(stamp) in valid:
            duplicate += 1
        valid[int(stamp)] = bar
    result = {"status": "unavailable", "signal": False, "regular_bars": len(valid),
              "missing_regular_minutes": (close_ms - open_ms) // MINUTE_MS - len(valid),
              "invalid_bars": invalid, "duplicate_timestamps": duplicate}
    if duplicate:
        return {**result, "reason": "duplicate_minute_timestamps"}
    if invalid:
        return {**result, "reason": "invalid_or_nonfinite_minute_data"}
    if not positive(anchor):
        return {**result, "reason": "missing_or_invalid_anchor"}
    if open_ms not in valid:
        return {**result, "reason": "missing_session_first_minute"}
    start_ms = max(open_ms, int(publication.timestamp() * 1000)) if publication else open_ms
    noon = opened.astimezone(ET).replace(hour=12, minute=0, second=0, microsecond=0)
    deadline = min(close_ms, int(noon.timestamp() * 1000)) if before_noon else close_ms
    eligible = [s for s in sorted(valid) if s >= start_ms and s + MINUTE_MS < deadline]
    signal = next((s for s in eligible if valid[s]["c"] <= anchor * (1 - PROTOCOL["threshold_fraction"])), None)
    if signal is None:
        return {**result, "status": "no_entry", "reason": "no_observed_completed_bar_trigger_before_deadline"}
    result.update(signal=True, signal_at=datetime.fromtimestamp(signal / 1000, ET).isoformat(),
                  signal_observed_at=datetime.fromtimestamp((signal + MINUTE_MS) / 1000, ET).isoformat(),
                  signal_close=valid[signal]["c"], threshold_price=anchor * 0.96)
    entry = signal + MINUTE_MS
    if entry not in valid:
        return {**result, "reason": "missing_immediately_next_entry_minute"}
    entry_price = valid[entry]["o"]
    result.update(entry_at=datetime.fromtimestamp(entry / 1000, ET).isoformat(), entry_price=entry_price,
                  entry_return_from_anchor=entry_price / anchor - 1)
    if close_ms - MINUTE_MS not in valid:
        return {**result, "reason": "missing_expected_final_exit_minute"}
    exit_price = valid[close_ms - MINUTE_MS]["c"]
    price_return = exit_price / entry_price - 1
    result.update(status="completed", reason=None, exit_price=exit_price,
                  exit_at=datetime.fromtimestamp((close_ms - MINUTE_MS) / 1000, ET).isoformat(),
                  exit_close_observed_at=closed.astimezone(ET).isoformat(),
                  price_return=price_return,
                  matched_open_return=exit_price / valid[open_ms]["o"] - 1)
    return result


def summarize(rows):
    done = [r for r in rows if r["status"] == "completed"]
    no_entry = sum(r["status"] == "no_entry" for r in rows)
    out = {"candidates": len(rows), "signals": sum(r["signal"] for r in rows),
           "completed": len(done), "no_entry": no_entry,
           "unavailable": sum(r["status"] == "unavailable" for r in rows),
           "unavailable_reasons": dict(Counter(r["reason"] for r in rows if r["status"] == "unavailable"))}
    values = [r["price_return"] for r in done]
    rng = random.Random(PROTOCOL["bootstrap"]["seed"])
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(PROTOCOL["bootstrap"]["resamples"])) if values else []
    out["price_returns"] = {
        "mean_return": statistics.fmean(values) if values else None,
        "median_return": statistics.median(values) if values else None,
        "wins": sum(v > 0 for v in values), "losses": sum(v < 0 for v in values),
        "win_rate": sum(v > 0 for v in values) / len(values) if values else None,
        "worst_return": min(values) if values else None,
        "bootstrap_mean_95pct_interval": [means[249], means[9749]] if means else None,
        "known_opportunity_mean_return": sum(values) / (len(done) + no_entry) if len(done) + no_entry else None,
        "known_opportunity_denominator": len(done) + no_entry,
    }
    return out


def analyze():
    rows = json.loads((ROOT / "cohort.json").read_text())
    daily = {(r["ticker"], r["date"]): r.get("payload", {}) for r in json.loads((ROOT / "daily_prices.json").read_text())}
    minute = {(r["ticker"], r["session"]): r for r in json.loads((ROOT / "minute_manifest.json").read_text())}
    old_fetch = {(r["ticker"], r["entry_date"]): r for r in json.loads((OLD / "minute_fetch_manifest.json").read_text())}
    split_metadata = json.loads((ROOT / "sources/split_metadata.json").read_text())
    cases = []
    for row in rows:
        path = REPO / minute[(row["ticker"], row["session"])]["path"]
        assert digest(path) == minute[(row["ticker"], row["session"])]["sha256"]
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
            minute_doc = json.load(handle)
        assert minute_doc["ticker"] == row["ticker"] and minute_doc["entry_date"] == row["session"] and minute_doc["adjusted"] is False
        if row["session"] != "2026-11-09":
            assert old_fetch[(row["ticker"], row["session"])]["ok"] is True
        bars = minute_doc["bars"]
        opened, closed = (datetime.fromisoformat(row[k]) for k in ("open_at", "close_at"))
        current = daily.get((row["ticker"], row["session"]), {})
        previous = daily.get((row["ticker"], row["previous_session"]), {})
        for anchor_name, payload, field, expected_day in (
            ("session_open", current, "open", row["session"]),
            ("previous_close", previous, "close", row["previous_session"]),
        ):
            anchor = payload.get(field) if payload.get("status") == "OK" and payload.get("from") == expected_day and payload.get("symbol") == row["ticker"] else None
            anchor_issue = None
            if anchor_name == "previous_close":
                if not split_metadata["complete"]:
                    anchor_issue = "split_basis_not_verified"
                elif any(r["ticker"] == row["ticker"] and row["previous_session"] < r["execution_date"] <= row["session"] for r in split_metadata["relevant_events"]):
                    anchor_issue = "split_crosses_previous_close_reference"
            for availability in ("hypothetical_at_open", "publication_time_sensitivity"):
                publication = datetime.fromisoformat(row["published_at_et"]) if row["published_at_et"] else None
                for window in ("before_noon", "whole_session"):
                    base = {"session": row["session"], "ticker": row["ticker"], "anchor": anchor_name,
                            "anchor_price": anchor, "availability": availability, "window": window,
                            "published_at_et": row["published_at_et"]}
                    if anchor_issue:
                        result = {"status": "unavailable", "reason": anchor_issue, "signal": False}
                    elif minute_doc.get("error_type"):
                        result = {"status": "unavailable", "reason": "minute_request_failed", "signal": False}
                    elif availability == "publication_time_sensitivity" and publication is None:
                        result = {"status": "unavailable", "reason": "publication_time_unknown", "signal": False}
                    else:
                        result = evaluate(bars, anchor, opened, closed,
                                          publication if availability == "publication_time_sensitivity" else None,
                                          before_noon=window == "before_noon")
                    cases.append({**base, **result})
    summaries = []
    for anchor in ("session_open", "previous_close"):
        for availability in ("hypothetical_at_open", "publication_time_sensitivity"):
            for window in ("before_noon", "whole_session"):
                chosen = [r for r in cases if (r["anchor"], r["availability"], r["window"]) == (anchor, availability, window)]
                summaries.append({"anchor": anchor, "availability": availability, "window": window, **summarize(chosen)})
    dump(ROOT / "results.json", {"protocol_sha256": digest(ROOT / "protocol.json"),
         "analysis_amendment_sha256": digest(ROOT / "analysis_amendment.json"),
         "return_definition": ANALYSIS_AMENDMENT["active_return_definition"],
         "modeled_trading_cost_fraction": 0,
         "cohort_sha256": digest(ROOT / "cohort.json"),
         "minute_manifest_sha256": digest(ROOT / "minute_manifest.json"),
         "script_sha256": digest(Path(__file__)),
         "clarification_sha256": digest(ROOT / "protocol_clarification.json"),
         "daily_prices_sha256": digest(ROOT / "daily_prices.json"),
         "split_metadata_sha256": digest(ROOT / "sources/split_metadata.json"),
         "generated_at_utc": datetime.now(timezone.utc).isoformat(), "summaries": summaries, "cases": cases})
    for row in summaries:
        print(json.dumps({k: row[k] for k in ("anchor", "availability", "window", "candidates", "signals", "completed", "no_entry", "unavailable", "price_returns")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fetch", "splits", "analyze"))
    action = parser.parse_args().action
    repo_imports()
    {"fetch": fetch_sources, "splits": fetch_split_metadata, "analyze": analyze}[action]()
