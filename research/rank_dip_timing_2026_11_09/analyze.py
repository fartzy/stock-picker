"""Frozen, zero-expense Rank #1 dip-timing study; no network or production writes.

Minute bars use the provider's documented start timestamps. A closing price is
observed at the end of that minute. Fixed checkpoints and first-event entries
use the next bar's actual open, never a hindsight low or same-bar threshold fill.
https://massive.com/docs/rest/stocks/aggregates/custom-bars
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ET = ZoneInfo("America/New_York")
MINUTE = 60_000
PROTOCOL = json.loads((ROOT / "analysis_protocol.json").read_text())
if PROTOCOL["expense_fraction"] != 0:
    raise ValueError("The study must use zero expense and unchanged price returns")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dump_gzip(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def ms(value):
    return int(value.timestamp() * 1000)


def at_clock(opened, clock):
    hour, minute = map(int, clock.split(":"))
    return opened.astimezone(ET).replace(hour=hour, minute=minute, second=0, microsecond=0)


def iso(stamp):
    return datetime.fromtimestamp(stamp / 1000, ET).isoformat()


def week_id(session):
    year, week, _ = datetime.fromisoformat(session).isocalendar()
    return f"{year}-W{week:02d}"


def time_bin(stamp):
    value = datetime.fromtimestamp(stamp / 1000, ET)
    minute = value.hour * 60 + value.minute
    for end, name in ((600, "09:30-10:00"), (630, "10:00-10:30"),
                      (660, "10:30-11:00"), (720, "11:00-12:00"),
                      (840, "12:00-14:00")):
        if minute < end:
            return name
    return "14:00-close"


def depth_band(depth):
    if depth < PROTOCOL["threshold_fraction"] - 1e-12:
        return "below4pct"
    return "4-6pct" if depth < .06 else "6-10pct" if depth < .10 else "10pct_plus"


def prepare_session(bars, opened, closed):
    open_ms, close_ms = ms(opened), ms(closed)
    valid = {}
    invalid = duplicate = 0
    for bar in bars:
        if not isinstance(bar, dict):
            invalid += 1
            continue
        stamp = bar.get("t")
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool) or not math.isfinite(stamp) or stamp % MINUTE:
            invalid += 1
            continue
        if not open_ms <= stamp < close_ms:
            continue
        values = [bar.get(k) for k in ("o", "h", "l", "c")]
        if not all(positive(v) for v in values) or not values[2] <= min(values[0], values[3]) <= max(values[0], values[3]) <= values[1]:
            invalid += 1
            continue
        if int(stamp) in valid:
            duplicate += 1
        valid[int(stamp)] = bar
    info = {"regular_bars": len(valid), "expected_regular_minutes": (close_ms - open_ms) // MINUTE,
            "missing_regular_minutes": (close_ms - open_ms) // MINUTE - len(valid),
            "invalid_bars": invalid, "duplicate_timestamps": duplicate}
    reason = ("invalid_or_nonfinite_minute_data" if invalid else "duplicate_minute_timestamps" if duplicate
              else "missing_session_first_minute" if open_ms not in valid else None)
    return {"bars": valid, "open_ms": open_ms, "close_ms": close_ms,
            "opened": opened, "closed": closed, "coverage": info, "reason": reason}


def context(session, anchor, anchor_name, availability, official_open, observed_until):
    bars = session["bars"]
    threshold = anchor * (1 - PROTOCOL["threshold_fraction"])
    known = [t for t in sorted(bars) if t + MINUTE <= observed_until]
    first = next((t for t in known if bars[t]["c"] <= threshold), None)
    opening_gap = anchor_name == "previous_close" and positive(official_open) and official_open <= threshold
    observed_onset = session["open_ms"] if opening_gap else first + MINUTE if first is not None else None
    before = bars.get(ms(availability) - MINUTE)
    was_below = seen_below = bool(opening_gap)
    episode = session["open_ms"] if opening_gap else None
    recovered = False
    for stamp in known:
        below = bars[stamp]["c"] <= threshold
        if below and not was_below:
            episode = stamp + MINUTE
        if not below:
            recovered = recovered or seen_below
            episode = None
        seen_below = seen_below or below
        was_below = below
    return {"threshold_price": threshold,
            "opening_gap_already_below_threshold": bool(opening_gap),
            "first_completed_bar_threshold_at": iso(first + MINUTE) if first is not None else None,
            "first_observed_dip_at": iso(observed_onset) if observed_onset is not None else None,
            "onset_bin": "opening_gap" if opening_gap else time_bin(observed_onset) if observed_onset is not None else None,
            "already_below_threshold_at_availability": before["c"] <= threshold if before else None,
            "current_observed_dip_episode_at": iso(episode) if episode is not None else None,
            "recovered_above_threshold_since_first_observed_dip": recovered}


def base_result(session, anchor, anchor_name, availability, official_open):
    result = {"status": "unavailable", "signal": False, **session["coverage"]}
    if session["reason"]:
        return result, session["reason"]
    if not positive(anchor):
        return result, "missing_or_invalid_anchor"
    result.update(context(session, anchor, anchor_name, availability, official_open, ms(availability)))
    return result, None


def finish_entry(session, result, signal_stamp, entry_stamp, anchor):
    bars = session["bars"]
    result.update(signal_observed_at=iso(signal_stamp + MINUTE), signal_bar_at=iso(signal_stamp),
                  signal_close=bars[signal_stamp]["c"], signal_depth_fraction=1 - bars[signal_stamp]["c"] / anchor,
                  entry_at=iso(entry_stamp), entry_time_bin=time_bin(entry_stamp),
                  early_late="early_before10:30" if entry_stamp < ms(at_clock(session["opened"], "10:30")) else "late_10:30_plus",
                  minutes_from_open=(entry_stamp - session["open_ms"]) / MINUTE,
                  minutes_remaining=(session["close_ms"] - entry_stamp) / MINUTE)
    result["depth_band"] = depth_band(result["signal_depth_fraction"])
    onset = result.get("first_observed_dip_at")
    result["minutes_since_first_observed_dip"] = (entry_stamp - ms(datetime.fromisoformat(onset))) / MINUTE if onset else None
    episode = result.get("current_observed_dip_episode_at")
    result["minutes_since_current_observed_dip_episode"] = (entry_stamp - ms(datetime.fromisoformat(episode))) / MINUTE if episode else None
    if entry_stamp not in bars:
        return {**result, "reason": "missing_immediately_next_entry_minute"}
    price = bars[entry_stamp]["o"]
    result.update(entry_price=price, entry_return_from_anchor=price / anchor - 1)
    exit_stamp = session["close_ms"] - MINUTE
    if exit_stamp not in bars:
        return {**result, "reason": "missing_expected_final_exit_minute"}
    exit_price = bars[exit_stamp]["c"]
    remaining = [b["l"] for t, b in bars.items() if entry_stamp <= t <= exit_stamp]
    result.update(status="completed", reason=None, exit_price=exit_price,
                  exit_bar_at=iso(exit_stamp), exit_observed_at=iso(session["close_ms"]),
                  price_return=exit_price / price - 1,
                  worst_observed_further_return=min(remaining) / price - 1,
                  matched_open_return=exit_price / bars[session["open_ms"]]["o"] - 1)
    return result


def first_event(session, anchor, anchor_name, availability, official_open, before_noon=False):
    result, issue = base_result(session, anchor, anchor_name, availability, official_open)
    if issue:
        return {**result, "reason": issue}
    deadline = min(session["close_ms"], ms(at_clock(session["opened"], "12:00"))) if before_noon else session["close_ms"]
    eligible = [t for t in sorted(session["bars"]) if ms(availability) <= t + MINUTE < deadline]
    signal = next((t for t in eligible if session["bars"][t]["c"] <= result["threshold_price"]), None)
    if signal is None:
        return {**result, "status": "no_entry", "reason": "no_observed_completed_bar_trigger_before_deadline"}
    result["signal"] = True
    result.update(context(session, anchor, anchor_name, availability, official_open, signal + MINUTE))
    return finish_entry(session, result, signal, signal + MINUTE, anchor)


def checkpoint(session, anchor, anchor_name, availability, official_open, clock):
    result, issue = base_result(session, anchor, anchor_name, availability, official_open)
    if issue:
        return {**result, "reason": issue, "dip": None}
    entry = ms(at_clock(session["opened"], clock))
    if entry < ms(availability):
        return {**result, "reason": "ranking_not_yet_available", "dip": None}
    if not session["open_ms"] < entry < session["close_ms"]:
        return {**result, "reason": "checkpoint_outside_regular_session", "dip": None}
    signal = entry - MINUTE
    if signal not in session["bars"]:
        return {**result, "reason": "missing_exact_checkpoint_signal_minute", "dip": None}
    result.update(context(session, anchor, anchor_name, availability, official_open, entry))
    result["dip"] = result["signal"] = session["bars"][signal]["c"] <= result["threshold_price"]
    return finish_entry(session, result, signal, entry, anchor)


def interval_from_values(values):
    values = sorted(values)
    if not values:
        return None
    n = len(values)
    return [values[max(0, math.ceil(.025 * n) - 1)], values[min(n - 1, math.ceil(.975 * n) - 1)]]


def weekly_mean_interval(rows):
    done = [r for r in rows if r["status"] == "completed"]
    clusters = defaultdict(lambda: [0.0, 0])
    for row in done:
        pair = clusters[week_id(row["session"])]
        pair[0] += row["price_return"]
        pair[1] += 1
    rules = PROTOCOL["uncertainty"]
    if len(done) < rules["minimum_completed"] or len(clusters) < rules["minimum_completed_weeks"]:
        return {"interval": None, "reason": "too_few_completed_entries_or_weeks", "completed_weeks": len(clusters)}
    pairs = list(clusters.values())
    rng = random.Random(rules["seed"])
    means = []
    for _ in range(rules["replicas"]):
        drawn = rng.choices(pairs, k=len(pairs))
        means.append(sum(p[0] for p in drawn) / sum(p[1] for p in drawn))
    return {"interval": interval_from_values(means), "reason": None, "completed_weeks": len(clusters)}


def summarize(rows, include_cash=False):
    done = [r for r in rows if r["status"] == "completed"]
    prices = [r["price_return"] for r in done]
    no_entry = sum(r["status"] == "no_entry" for r in rows)
    out = {"observations": len(rows), "unique_stock_days": len({(r["session"], r["ticker"]) for r in rows}),
           "candidate_sessions": len({r["session"] for r in rows}), "completed": len(done),
           "completed_stock_days": len({(r["session"], r["ticker"]) for r in done}),
           "completed_sessions": len({r["session"] for r in done}),
           "signals": sum(r.get("signal", False) for r in rows), "no_entry": no_entry,
           "unavailable": sum(r["status"] == "unavailable" for r in rows),
           "unavailable_reasons": dict(Counter(r["reason"] for r in rows if r["status"] == "unavailable")),
           "mean_return": statistics.fmean(prices) if prices else None,
           "median_return": statistics.median(prices) if prices else None,
           "wins": sum(x > 0 for x in prices), "losses": sum(x < 0 for x in prices),
           "flats": sum(x == 0 for x in prices), "win_rate": sum(x > 0 for x in prices) / len(prices) if prices else None,
           "worst_return": min(prices) if prices else None,
           "mean_observed_further_downside": statistics.fmean(r["worst_observed_further_return"] for r in done) if done else None,
           "worst_observed_further_return": min(r["worst_observed_further_return"] for r in done) if done else None,
           "weekly_mean_95pct": weekly_mean_interval(rows)}
    if include_cash:
        denominator = len(done) + no_entry
        out.update(known_opportunity_denominator=denominator,
                   known_opportunity_mean_return=sum(prices) / denominator if denominator else None)
    return out


def difference_interval(left, right):
    """Resample weeks jointly for both groups; never resample individual clocks."""
    left = [r for r in left if r["status"] == "completed"]
    right = [r for r in right if r["status"] == "completed"]
    result = {"left_completed": len(left), "right_completed": len(right), "mean_difference": None,
              "weekly_95pct_interval": None, "reason": None}
    if not left or not right:
        return {**result, "reason": "one_group_has_no_completed_entries"}
    result["mean_difference"] = statistics.fmean(r["price_return"] for r in left) - statistics.fmean(r["price_return"] for r in right)
    rules = PROTOCOL["uncertainty"]
    clusters = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for rows, offset in ((left, 0), (right, 2)):
        for row in rows:
            pair = clusters[week_id(row["session"])]
            pair[offset] += row["price_return"]
            pair[offset + 1] += 1
    result["weeks"] = len(clusters)
    if min(len(left), len(right)) < rules["minimum_completed"] or min(len({week_id(r["session"]) for r in group}) for group in (left, right)) < rules["minimum_completed_weeks"]:
        return {**result, "reason": "too_few_entries_or_weeks_in_one_group"}
    rng = random.Random(rules["seed"])
    pairs = list(clusters.values())
    values = []
    for _ in range(rules["replicas"]):
        drawn = rng.choices(pairs, k=len(pairs))
        n1, n2 = sum(p[1] for p in drawn), sum(p[3] for p in drawn)
        if n1 and n2:
            values.append(sum(p[0] for p in drawn) / n1 - sum(p[2] for p in drawn) / n2)
    result["bootstrap_valid_replicas"] = len(values)
    result["weekly_95pct_interval"] = interval_from_values(values)
    return result


def cohort_label(row):
    return "live_rank1" if row["cohort"] == "live" and row["rank"] == 1 else "live_ranks2_5" if row["cohort"] == "live" else row["cohort"] + "_rank1"


def anchor_payload(daily, ticker, day, field):
    record = daily.get((ticker, day), {})
    payload = record.get("payload", {})
    provider_ticker = record.get("provider_ticker", ticker)
    if provider_ticker != ticker and not record.get("provider_alias_verified"):
        return None
    if record.get("adjusted") is not False or payload.get("status") != "OK" or payload.get("symbol") != provider_ticker or payload.get("from") != day:
        return None
    return payload.get(field) if positive(payload.get(field)) else None


def split_issue(metadata, row):
    if not metadata.get("complete"):
        return "split_basis_not_verified"
    if metadata.get("coverage_start", row["previous_session"]) > row["previous_session"] or metadata.get("coverage_end", row["session"]) < row["session"]:
        return "split_metadata_outside_coverage"
    events = metadata.get("relevant_events", metadata.get("events", []))
    if any(event["ticker"] == row["ticker"] and row["previous_session"] < event["execution_date"] <= row["session"] for event in events):
        return "split_crosses_previous_close_reference"
    return None


def load_sources():
    progress_path = ROOT / "acquisition_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("status") not in ("complete", "completed") and progress.get("complete") is not True:
            raise ValueError("Acquisition is not marked complete; refusing a partial-data result")
    cohort = json.loads((ROOT / "cohort.json").read_text())
    daily = {(r["ticker"], r["date"]): r for r in json.loads((ROOT / "daily_prices.json").read_text())}
    minute = {(r["ticker"], r["session"]): r for r in json.loads((ROOT / "minute_manifest.json").read_text())}
    metadata_path = ROOT / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"complete": False}
    documents = {}
    for key, record in minute.items():
        if not record.get("path"):
            documents[key] = {"error_type": "missing_source_path"}
            continue
        path = REPO / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Minute source hash changed: {path}")
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as handle:
            doc = json.load(handle)
        if doc.get("ticker") != key[0] or doc.get("adjusted") is not False:
            raise ValueError(f"Minute source identity/basis mismatch: {path}")
        documents[key] = doc
        if record.get("complete") is False:
            documents[key] = {**doc, "error_type": "incomplete_minute_source"}
    return cohort, daily, documents, metadata


def create_cases(cohort, daily, documents, metadata):
    events, snapshots = [], []
    availability_clocks = [PROTOCOL["availability"]["primary_et"], *PROTOCOL["availability"]["event_sensitivities_et"]]
    for row in cohort:
        opened, closed = (datetime.fromisoformat(row[k]) for k in ("open_at", "close_at"))
        doc = documents.get((row["ticker"], row["session"]), {"error_type": "missing_minute_source"})
        session = prepare_session(doc.get("bars", []), opened, closed)
        open_price = anchor_payload(daily, row["ticker"], row["session"], "open")
        previous = anchor_payload(daily, row["ticker"], row["previous_session"], "close")
        first_bar = session["bars"].get(session["open_ms"])
        open_difference = first_bar["o"] / open_price - 1 if first_bar and positive(open_price) else None
        basis_conflict = open_difference is not None and abs(open_difference) > PROTOCOL["source_open_consistency"]["maximum_relative_disagreement"]
        common = {"cohort": cohort_label(row), "session": row["session"], "ticker": row["ticker"],
                  "rank": row["rank"], "rank_score": row["rank_score"],
                  "recorded_run_at_et": row.get("recorded_run_at_et", row.get("published_at_et")),
                  "opening_price": open_price, "minute_open_vs_official_relative_difference": open_difference}
        for anchor_name, anchor in (("session_open", open_price), ("previous_close", previous)):
            source_issue = ("minute_request_failed" if doc.get("error_type") else "daily_minute_open_price_basis_conflict" if basis_conflict
                            else split_issue(metadata, row) if anchor_name == "previous_close" else None)
            for clock in availability_clocks:
                availability = at_clock(opened, clock)
                for window in PROTOCOL["event_windows"]:
                    base = {**common, "anchor": anchor_name, "anchor_price": anchor, "availability_et": clock, "window": window}
                    result = {"status": "unavailable", "signal": False, "reason": source_issue} if source_issue else first_event(session, anchor, anchor_name, availability, open_price, window == "before_noon")
                    events.append({**base, **result})
            availability = at_clock(opened, PROTOCOL["availability"]["primary_et"])
            for clock in PROTOCOL["checkpoints_et"]:
                base = {**common, "anchor": anchor_name, "anchor_price": anchor, "availability_et": PROTOCOL["availability"]["primary_et"], "checkpoint_et": clock}
                result = {"status": "unavailable", "signal": False, "dip": None, "reason": source_issue} if source_issue else checkpoint(session, anchor, anchor_name, availability, open_price, clock)
                snapshots.append({**base, **result})
    return events, snapshots


def summarize_cases(events, snapshots):
    event_summaries, timing, differences, checkpoints = [], [], [], []
    cohorts = sorted({r["cohort"] for r in events})
    for cohort in cohorts:
        for anchor in PROTOCOL["anchors"]:
            chosen = [r for r in events if r["cohort"] == cohort and r["anchor"] == anchor]
            for availability in [PROTOCOL["availability"]["primary_et"], *PROTOCOL["availability"]["event_sensitivities_et"]]:
                for window in PROTOCOL["event_windows"]:
                    subset = [r for r in chosen if r["availability_et"] == availability and r["window"] == window]
                    event_summaries.append({"cohort": cohort, "anchor": anchor, "availability_et": availability, "window": window, **summarize(subset, True)})
            primary = [r for r in chosen if r["availability_et"] == PROTOCOL["availability"]["primary_et"] and r["window"] == "whole_session"]
            for dimension, bins in (("entry_time_bin", PROTOCOL["entry_and_onset_bins_et"]), ("onset_bin", ["opening_gap", *PROTOCOL["entry_and_onset_bins_et"]]), ("depth_band", PROTOCOL["signal_depth_bands"])):
                for value in bins:
                    subset = [r for r in primary if r.get(dimension) == value and r.get("signal")]
                    timing.append({"cohort": cohort, "anchor": anchor, "dimension": dimension, "value": value, **summarize(subset)})
            for band in ["all_depths", *PROTOCOL["signal_depth_bands"]]:
                selected = [r for r in primary if band == "all_depths" or r.get("depth_band") == band]
                early = [r for r in selected if r.get("early_late") == "early_before10:30"]
                late = [r for r in selected if r.get("early_late") == "late_10:30_plus"]
                differences.append({"cohort": cohort, "anchor": anchor, "comparison": "early_minus_late", "depth_band": band,
                                    "early": summarize(early), "late": summarize(late), **difference_interval(early, late)})
            for clock in PROTOCOL["checkpoints_et"]:
                selected = [r for r in snapshots if r["cohort"] == cohort and r["anchor"] == anchor and r["checkpoint_et"] == clock]
                groups = {"all_eligible": selected, "currently_down_at_least4pct": [r for r in selected if r.get("dip") is True], "not_currently_down4pct": [r for r in selected if r.get("dip") is False]}
                for name, group in groups.items():
                    checkpoints.append({"cohort": cohort, "anchor": anchor, "checkpoint_et": clock, "group": name, **summarize(group)})
                differences.append({"cohort": cohort, "anchor": anchor, "comparison": "same_clock_dip_minus_nondip", "checkpoint_et": clock,
                                    **difference_interval(groups["currently_down_at_least4pct"], groups["not_currently_down4pct"])})
    return {"first_event_summaries": event_summaries, "timing_summaries": timing,
            "checkpoint_summaries": checkpoints, "comparisons": differences}


def main():
    cohort, daily, documents, metadata = load_sources()
    events, snapshots = create_cases(cohort, daily, documents, metadata)
    for cohort_name in PROTOCOL["primary_cohorts"]:
        for anchor in PROTOCOL["anchors"]:
            group = [r for r in events if r["cohort"] == cohort_name and r["anchor"] == anchor
                     and r["availability_et"] == "09:35" and r["window"] == "whole_session" and r["status"] == "completed"]
            brief = {"stage": "point_estimates_before_cluster_review", "cohort": cohort_name, "anchor": anchor,
                     "completed": len(group), "wins": sum(r["price_return"] > 0 for r in group),
                     "mean_return": statistics.fmean(r["price_return"] for r in group) if group else None}
            for period in ("early_before10:30", "late_10:30_plus"):
                values = [r["price_return"] for r in group if r["early_late"] == period]
                brief[period] = {"n": len(values), "mean": statistics.fmean(values) if values else None}
            print(json.dumps(brief), flush=True)
    summaries = summarize_cases(events, snapshots)
    source_names = ["analysis_protocol.json", "cohort.json", "daily_prices.json", "minute_manifest.json", "analyze.py"]
    if (ROOT / "split_metadata.json").exists():
        source_names.append("split_metadata.json")
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "expense_fraction": 0,
              "return_definition": PROTOCOL["return_definition"],
              "source_hashes": {name: digest(ROOT / name) for name in source_names},
              "cohort_memberships": len(cohort), "unique_stock_days": len({(r["session"], r["ticker"]) for r in cohort}),
              "unique_sessions": len({r["session"] for r in cohort}), **summaries}
    for name, cases in (("event_cases", events), ("checkpoint_cases", snapshots)):
        path = ROOT / f"{name}.json.gz"
        dump_gzip(path, cases)
        result[name] = {"path": str(path.relative_to(REPO)), "sha256": digest(path), "records": len(cases)}
    dump(ROOT / "results.json", result)
    for row in summaries["first_event_summaries"]:
        if row["availability_et"] == "09:35" and row["window"] == "before_noon":
            print(json.dumps({k: row[k] for k in ("cohort", "anchor", "completed", "wins", "mean_return", "unavailable", "weekly_mean_95pct")}))


if __name__ == "__main__":
    main()
