"""Offline paired-entry addendum; leave every original study artifact unchanged.

Recompute identical-stock/day returns directly from raw bars, then compare buying
at the assumed 09:35 ET ranking availability with waiting for the frozen 4% rule.
No original analyzer import, network request, price adjustment, or expense model.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ET = ZoneInfo("America/New_York")
MINUTE = 60_000
COHORTS = ("live_rank1", "research_baseline_rank1", "research_all_dips_rank1")
LABELS = dict(zip(COHORTS, ("Actual saved live Rank #1", "Historical baseline Rank #1", "Historical all-21-dip-features Rank #1")))
FROZEN = ("analysis_protocol.json", "analyze.py", "test_analysis.py", "README.md", "results.json",
          "event_cases.json.gz", "checkpoint_cases.json.gz", "audit_checks.json", "independent_review.json",
          "cohort.json", "cohort_provenance.json", "daily_prices.json", "minute_manifest.json",
          "split_metadata.json", "acquisition_audit.json")


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    return round(datetime.fromisoformat(value).timestamp() * 1000)


def positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def assert_close(left, right):
    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


def cohort_label(row):
    return "live_rank1" if row["cohort"] == "live" else row["cohort"] + "_rank1"


def regular_bars(document, opened, closed):
    bars = {}
    for bar in document.get("bars", []):
        t = bar["t"]
        if not opened <= t < closed:
            continue
        assert t % MINUTE == 0 and t not in bars
        assert all(positive(bar[k]) for k in ("o", "h", "l", "c"))
        assert bar["l"] <= min(bar["o"], bar["c"]) <= max(bar["o"], bar["c"]) <= bar["h"]
        bars[t] = bar
    return bars


def paired_summary(rows, benchmark_key, waiting_key):
    differences = [r[waiting_key] - r[benchmark_key] for r in rows]
    return {
        "matched_days": len(rows),
        "benchmark_mean_return": statistics.fmean(r[benchmark_key] for r in rows) if rows else None,
        "waiting_mean_return": statistics.fmean(r[waiting_key] for r in rows) if rows else None,
        "mean_paired_difference": statistics.fmean(differences) if rows else None,
        "waiting_better_days": sum(x > 1e-12 for x in differences),
        "equal_days": sum(abs(x) <= 1e-12 for x in differences),
        "waiting_worse_days": sum(x < -1e-12 for x in differences),
    }


def compute():
    frozen_hashes = {name: sha(ROOT / name) for name in FROZEN}
    protocol, results = read(ROOT / "analysis_protocol.json"), read(ROOT / "results.json")
    assert protocol["expense_fraction"] == results["expense_fraction"] == 0
    assert protocol["threshold_fraction"] == .04 and protocol["availability"]["primary_et"] == "09:35"
    for name, expected in results["source_hashes"].items():
        assert sha(ROOT / name) == expected, name
    event_manifest = results["event_cases"]
    assert sha(REPO / event_manifest["path"]) == event_manifest["sha256"]
    selected = [r for r in read(REPO / event_manifest["path"]) if r["cohort"] in COHORTS
                and r["anchor"] == "session_open" and r["availability_et"] == "09:35" and r["window"] == "whole_session"]
    cohort = {(cohort_label(r), r["session"], r["ticker"]): r for r in read(ROOT / "cohort.json") if r["rank"] == 1}
    assert len(selected) == len(cohort) == 721
    manifests = {(r["ticker"], r["session"]): r for r in read(ROOT / "minute_manifest.json")}
    documents, dip_rows, policy_rows = {}, [], []
    for row in selected:
        key = (row["cohort"], row["session"], row["ticker"])
        source = cohort[key]
        market_key = (row["ticker"], row["session"])
        manifest = manifests[market_key]
        if market_key not in documents:
            path = REPO / manifest["path"]
            assert sha(path) == manifest["sha256"] and manifest["adjusted"] is False
            document = read(path)
            assert document["ticker"] == row["ticker"] and document["adjusted"] is False
            documents[market_key] = document
        opened, closed = stamp(source["open_at"]), stamp(source["close_at"])
        availability = datetime.fromisoformat(source["open_at"]).astimezone(ET).replace(hour=9, minute=35)
        benchmark_at, exit_at = round(availability.timestamp() * 1000), closed - MINUTE
        bars = regular_bars(documents[market_key], opened, closed)
        common = {"cohort": row["cohort"], "session": row["session"], "ticker": row["ticker"],
                  "saved_event_status": row["status"], "saved_event_reason": row.get("reason"),
                  "source_path": manifest["path"], "source_sha256": manifest["sha256"]}
        dip_return = None
        if row["status"] in ("completed", "no_entry"):
            assert opened in bars and positive(row["anchor_price"])
            assert_close(bars[opened]["o"], row["anchor_price"])
            signals = [t for t in sorted(bars) if benchmark_at <= t + MINUTE < closed and bars[t]["c"] <= row["anchor_price"] * .96]
            if row["status"] == "no_entry":
                assert not signals
            else:
                assert signals and signals[0] + MINUTE == stamp(row["entry_at"])
                entry_at = signals[0] + MINUTE
                assert entry_at in bars and exit_at in bars
                opening, entry, closing = bars[opened]["o"], bars[entry_at]["o"], bars[exit_at]["c"]
                open_return, dip_return = closing / opening - 1, closing / entry - 1
                assert_close(entry, row["entry_price"])
                assert_close(closing, row["exit_price"])
                assert_close(dip_return, row["price_return"])
                assert_close(open_return, row["matched_open_return"])
                assert_close(dip_return - open_return, closing * (1 / entry - 1 / opening))
                assert entry >= opening or dip_return > open_return
                dip_rows.append({**common, "open_price": opening, "dip_entry_price": entry,
                                 "dip_entry_at": row["entry_at"], "identical_exit_price": closing,
                                 "open_return": open_return, "dip_return": dip_return,
                                 "paired_difference": dip_return - open_return,
                                 "entry_below_open": entry < opening,
                                 "next_open_rebounded_above_4pct_threshold": entry > opening * .96})
        reasons = []
        if row["status"] == "unavailable":
            reasons.append("wait_rule_unavailable:" + row["reason"])
        if benchmark_at not in bars:
            reasons.append("missing_09:35_benchmark_entry_minute")
        if exit_at not in bars:
            reasons.append("missing_expected_final_exit_minute")
        matched = not reasons
        policy = {**common, "matched": matched, "exclusion_reasons": reasons,
                  "benchmark_entry_at": availability.isoformat(), "benchmark_open_price": bars.get(benchmark_at, {}).get("o"),
                  "identical_exit_price": bars.get(exit_at, {}).get("c"),
                  "prior_09:34_signal_bar_present": benchmark_at - MINUTE in bars,
                  "buy_09:35_return": None, "waiting_return": None}
        if matched:
            assert row["status"] in ("completed", "no_entry")
            policy["buy_09:35_return"] = bars[exit_at]["c"] / bars[benchmark_at]["o"] - 1
            policy["waiting_return"] = dip_return if row["status"] == "completed" else 0.0
        policy_rows.append(policy)
    summaries = []
    for name in COHORTS:
        dips = [r for r in dip_rows if r["cohort"] == name]
        opportunities = [r for r in policy_rows if r["cohort"] == name]
        matched = [r for r in opportunities if r["matched"]]
        triggered = [r for r in matched if r["saved_event_status"] == "completed"]
        no_entry = [r for r in matched if r["saved_event_status"] == "no_entry"]
        excluded = [r for r in opportunities if not r["matched"]]
        assert len(matched) + len(excluded) == len(opportunities)
        assert len(triggered) + len(no_entry) == len(matched)
        assert all(r["waiting_return"] == 0 for r in no_entry)
        summaries.append({
            "cohort": name, "candidate_days": len(opportunities),
            "same_dip_days_open_comparison": {**paired_summary(dips, "open_return", "dip_return"),
                "entries_below_open": sum(r["entry_below_open"] for r in dips),
                "next_open_rebounds_above_threshold": sum(r["next_open_rebounded_above_4pct_threshold"] for r in dips)},
            "same_dip_days_09:35_comparison": paired_summary(triggered, "buy_09:35_return", "waiting_return"),
            "matched_policy_comparison": {**paired_summary(matched, "buy_09:35_return", "waiting_return"),
                "triggered_days": len(triggered), "observed_no_entry_days": len(no_entry), "excluded_days": len(excluded),
                "exclusion_reason_counts_overlapping": dict(Counter(reason for r in excluded for reason in r["exclusion_reasons"])),
                "included_without_09:34_bar": sum(not r["prior_09:34_signal_bar_present"] for r in matched)},
            "matched_no_entry_days": paired_summary(no_entry, "buy_09:35_return", "waiting_return"),
        })
    assert {name: sha(ROOT / name) for name in FROZEN} == frozen_hashes
    return {"scope": "Additive paired-return audit; original protocol, analyzer, results and audits stay frozen.",
            "expense_fraction": 0, "availability_et": "09:35", "availability_chicago": "08:35",
            "opening_comparison": "Same stock/day and identical final-minute exit; opening-price reference is not a feasible ranked trade before ranking availability.",
            "policy_comparison": "Buy exact09:35 minute open versus frozen first4% dip rule; match identical candidate days with both outcomes known; waiting earnszero only on observed no-entry days; unavailable excluded fromboth.",
            "return_convention": "Price ratios only; arithmetic equal-capital mean per matched day; no compounding or expense deductions; hypothetical minute execution, not broker fills.",
            "checks": {"original_artifacts_unchanged": True, "source_hashes_verified": True,
                       "completed_entries_reconciled_with_raw_bars": len(dip_rows),
                       "known_no_entry_signals_reconciled_with_raw_bars": sum(r["saved_event_status"] == "no_entry" for r in policy_rows),
                       "primary_candidate_days": len(policy_rows), "comparison_uses_identical_exits": True,
                       "unconditional_09:35_buy_does_not_require_09:34_close": True,
                       "every_lower_entry_improves_identical_exit_price_return": True},
            "frozen_artifact_sha256": frozen_hashes, "audit_script_sha256": sha(Path(__file__)),
            "summaries": summaries, "dip_day_pairs": dip_rows, "policy_day_pairs": policy_rows}


def pct(value):
    return f"{value * 100:+.3f}%"


def points(value):
    return f"{value * 100:+.3f} pp"


def report(result):
    text = ["# Paired audit: a lower entry does improve the same-stock return", "",
            "**Yes: buying the same stock lower, then selling at the same price, improves the return.** The earlier headline answered whether the dip entries were profitable and whether early dips beat later dips. It did not show the matched improvement over buying those same stocks at the open. This addendum supplies that comparison without changing the original results.", "",
            "For the identical closing price `C`, opening price `O`, and lower dip entry `D`, the improvement is `C / D − C / O = C × (1 / D − 1 / O) > 0`. A loss can become smaller without becoming a profit. All numbers below use **zero trading expense**.", "",
            "| Rank #1 cohort | Same dip days | Buy at open → same exit | Dip entry → same exit | Improvement |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["summaries"]:
        s = row["same_dip_days_open_comparison"]
        text.append(f"| {LABELS[row['cohort']]} | {s['matched_days']} | {pct(s['benchmark_mean_return'])} | {pct(s['waiting_mean_return'])} | {points(s['mean_paired_difference'])} |")
    text.extend(["", "Every completed dip entry was below its own session open, so every one improved that same-stock, same-exit return. The signal is a completed-minute close down at least 4%; execution uses the next minute's open. That open rebounded above the exact 4% threshold in 1 live, 12 baseline, and 9 all-21 cases. The observed execution prices are retained, not replaced with an assumed threshold fill.", ""])
    for ticker, day in (("XENE", "2026-09-18"), ("IONS", "2026-11-09")):
        r = next(x for x in result["dip_day_pairs"] if x["cohort"] == "live_rank1" and x["ticker"] == ticker and x["session"] == day)
        text.append(f"{ticker} on {day}: open ${r['open_price']:.2f}, dip entry ${r['dip_entry_price']:.2f}, identical exit ${r['identical_exit_price']:.2f}. Return improves from {pct(r['open_return'])} to {pct(r['dip_return'])}, or {points(r['paired_difference'])}.")
        text.append("")
    text.extend(["The opening-price comparison conditions on days that later dipped. It proves the entry-price arithmetic for those days; it does not include days where waiting never triggered. The session open is also a reference price, not an assumed executable ranked trade before ranking availability.", "",
                 "A feasible reference under the frozen assumption is buying at **8:35 a.m. Chicago / 9:35 a.m. ET**. On dip days where both that entry and the dip entry are observed:", "",
                 "| Rank #1 cohort | Matched dip days | Buy at 8:35 Chicago | Wait for dip | Difference |",
                 "| --- | ---: | ---: | ---: | ---: |"])
    for row in result["summaries"]:
        s = row["same_dip_days_09:35_comparison"]
        text.append(f"| {LABELS[row['cohort']]} | {s['matched_days']} | {pct(s['benchmark_mean_return'])} | {pct(s['waiting_mean_return'])} | {points(s['mean_paired_difference'])} |")
    text.extend(["", "A dip can already be signaled by 8:35, in which case both rules enter at the same price. The 8:35 price may already be below the session open, so this comparison differs from the opening-price comparison.", "",
                 "For the full waiting policy, both alternatives must use the **same candidate days**. Waiting earns zero on an observed no-dip day; an unavailable outcome excludes the day from both alternatives. The unconditional 8:35 buy requires that minute's open and the final exit bar, not an 8:34 closing signal.", "",
                 "| Rank #1 cohort | Matched days (dip / no entry) | Excluded unknown | Buy at 8:35 Chicago: mean/day | Wait: mean/day | Wait minus buy |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in result["summaries"]:
        s = row["matched_policy_comparison"]
        text.append(f"| {LABELS[row['cohort']]} | {s['matched_days']} ({s['triggered_days']} / {s['observed_no_entry_days']}) | {s['excluded_days']} | {pct(s['benchmark_mean_return'])} | {pct(s['waiting_mean_return'])} | {points(s['mean_paired_difference'])} |")
    text.extend(["", "In these matched samples, waiting improved the return on the dip days on average, but its average across all matched candidate days was lower because it also skipped no-dip days. These are arithmetic, equal-capital daily averages, with zero earned on observed waiting days; they are not compounded portfolio returns or proof of a future edge.", "",
                 "The baseline and all-21 research cohorts cover the same 342 dates and stay separate; the live cohort has 37 dates. These are the frozen historical research models, not the current production feature subset. Availability at 8:35 Chicago is an assumption after the opening price is known, not proof of every live day's first publication time. Entry and exit prices are minute-bar proxies, not guaranteed broker fills. This paired comparison answers a different question from whether earlier dips outperform later dips; no new timing rule or parameter search was introduced.", "",
                 "`paired_entry_audit.json` retains every matched pair, unknown exclusion, overlapping exclusion reason, original artifact hash, and raw source hash. Assertions independently reconcile the saved first trigger and next-minute entry with raw bars, identical exits, lower-entry arithmetic, counts, and unchanged original artifacts. The script uses only the standard library and frozen files:", "",
                 "```sh", "python3 research/rank_dip_timing_2026_11_09/paired_entry_audit.py", "```", ""])
    return "\n".join(text)


def main():
    result = compute()
    (ROOT / "paired_entry_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (ROOT / "paired_entry_comparison.md").write_text(report(result))
    print(json.dumps({"checks": result["checks"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
