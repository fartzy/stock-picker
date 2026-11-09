"""User-amended retrospective morning-dip screen; never writes production data.

Run from the repository root with its existing hermetic Python:
bazel-bin/python/stock_picker/api/main.runfiles/rules_python++python+python_3_12_aarch64-apple-darwin/bin/python3 research/morning_dip_patterns_2026_11_09/run_screen.py

The immutable protocol and input snapshot must exist before this script runs.
Return values are fractions (0.01 means 1%), not percentages.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNFILES = (ROOT / "bazel-bin/python/stock_picker/api/main.runfiles").resolve()
sys.path[:0] = [str(ROOT / "python"), *map(str, RUNFILES.glob("rules_python++pip+*/site-packages"))]

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def original_artifact(protocol: dict, name: str) -> bytes:
    """Read the immutable pre-amendment result without duplicating large files."""
    amendment = protocol["mrcy_inclusion_amendment"]
    path = Path(amendment["original_artifact_directory"]) / name
    result = subprocess.run(
        ["git", "show", f"{amendment['original_commit']}:{path.as_posix()}"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    assert hashlib.sha256(result).hexdigest() == amendment["original_artifact_sha256"][name]
    return result


def inclusion_comparison(protocol: dict, summaries: list[dict], mrcy_counts: dict[str, int]) -> dict:
    """Preserve every original/revised cohort and era, not just favorable changes."""
    original = json.loads(original_artifact(protocol, "results.json"))
    previous = {(row["cohort"], row["era"]): row for row in original["summaries"]}
    keys = ["rows", "tickers", "dates", "gross_mean_equal_date", "matched_excess_equal_date",
            "matched_excess_bootstrap", "net_mean_equal_date_by_roundtrip_bps",
            "gross_winsor01_equal_date", "matched_excess_winsor01_equal_date"]
    comparisons = []
    for row in summaries:
        before = previous[(row["cohort"], row["era"])]
        change = {key: row[key] - before[key] for key in ["rows", "tickers", "dates", "gross_mean_equal_date"]}
        change["matched_excess_equal_date"] = (
            row["matched_excess_equal_date"] - before["matched_excess_equal_date"]
            if row["matched_excess_equal_date"] is not None else None
        )
        comparisons.append({"cohort": row["cohort"], "era": row["era"],
                            "before_mrcy_excluded": {key: before[key] for key in keys},
                            "after_mrcy_included": {key: row[key] for key in keys}, "change": change})
    assert len(comparisons) == len(previous) == 68
    return {"protocol_version": protocol["protocol_version"], "unit": "fraction",
            "amendment": protocol["mrcy_inclusion_amendment"],
            "comparison_status": "User-directed inclusion sensitivity after original outcomes; revised primary screen includes all eligible MRCY history.",
            "mrcy_rows_by_full_period_cohort": mrcy_counts, "comparisons": comparisons}


def valid_bars(bars: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Recheck source OHLCV rather than trusting serialized validity masks."""
    finite = np.isfinite(bars).all(axis=1)
    positive = (bars > 0).all(axis=1)
    o, h, l, c, _ = bars.T
    within = (h + 1e-5 >= np.maximum(o, c)) & (l - 1e-5 <= np.minimum(o, c)) & (h >= l)
    return finite & positive & within, {
        "missing_or_nonfinite": int((~finite).sum()),
        "nonpositive_finite": int((finite & ~positive).sum()),
        "ohlc_range_violation_finite_positive": int((finite & positive & ~within).sum()),
    }


def features(bars: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    o, h, l, c, v = bars.T
    close = pd.Series(c)
    ret = close.pct_change(fill_method=None)
    lag_close = close.shift(1).to_numpy()
    prior_low = pd.Series(l).shift(1).rolling(252, min_periods=252).min().to_numpy()
    prior_high = pd.Series(h).shift(1).rolling(252, min_periods=252).max().to_numpy()
    previous251_low = pd.Series(l).shift(2).rolling(251, min_periods=251).min().to_numpy()
    return {
        "history_ok": pd.Series(valid).shift(1).rolling(252, min_periods=252).sum().to_numpy() == 252,
        "adv20": pd.Series(c * v).shift(1).rolling(20, min_periods=20).mean().to_numpy(),
        "r1": ret.shift(1).to_numpy(),
        "r2": ret.shift(2).to_numpy(),
        "prior_low": prior_low,
        "near_low": (lag_close / prior_low >= 1 - 1e-10) & (lag_close / prior_low <= 1.02 + 1e-10),
        "near_high": (lag_close / prior_high <= 1 + 1e-10) & (lag_close / prior_high >= .98 - 1e-10),
        "new_low": pd.Series(l).shift(1).to_numpy() <= previous251_low * (1 + 1e-10),
        "open_low_distance": o / prior_low - 1,
        "recovery_numerator": o - lag_close,
        "recovery_denominator": close.shift(3).to_numpy() - lag_close,
        "pre_shock_vol": ret.rolling(20, min_periods=20).std(ddof=1).shift(3).to_numpy(),
    }


def check_indexing() -> None:
    """Synthetic assertions protect the decision-time boundary and day weighting."""
    c = np.arange(1, 301, dtype=float) + 100
    bars = np.column_stack([c, c + 1, c - 1, c, np.full(300, 1e6)])
    f = features(bars, np.ones(300, dtype=bool))
    t = 280
    assert f["prior_low"][t] == bars[t - 252:t, 2].min()
    assert np.isclose(f["r1"][t], c[t - 1] / c[t - 2] - 1)
    assert np.isclose(f["r2"][t], c[t - 2] / c[t - 3] - 1)
    returns = c[1:] / c[:-1] - 1
    assert np.isclose(f["pre_shock_vol"][t], returns[t - 23:t - 3].std(ddof=1))
    changed = bars.copy()
    changed[t:, 1:4] *= 10
    later = features(changed, np.ones(300, dtype=bool))
    for field in ["r1", "r2", "prior_low", "near_low", "near_high", "new_low", "pre_shock_vol"]:
        assert np.isclose(f[field][t], later[field][t]), field
    assert np.isclose((.01 + .02 + .03) / 3, np.array([.01, .02, .03]).mean())


def week_keys(dates: pd.Index) -> np.ndarray:
    iso = pd.DatetimeIndex(dates).isocalendar()
    return (iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)).to_numpy()


def bootstrap_weekly(values: pd.Series, seed: int, replications: int) -> dict:
    """Resample whole active weeks; retain all dates and equal-date weighting."""
    values = values.dropna()
    if values.empty:
        return {"weeks": 0, "ci95": None, "sparse": True}
    frame = pd.DataFrame({"value": values.to_numpy(), "week": week_keys(values.index)})
    groups = frame.groupby("week").value.agg(["sum", "count"])
    n = len(groups)
    if n < 2:
        return {"weeks": n, "ci95": None, "sparse": True}
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, n, size=(replications, n))
    draws = groups["sum"].to_numpy()[sample].sum(axis=1) / groups["count"].to_numpy()[sample].sum(axis=1)
    return {"weeks": n, "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "sparse": n < 10}


def era_mask(dates: pd.Series, era: str) -> pd.Series:
    year = dates.dt.year
    return year <= 2024 if era == "2022_2024" else year == int(era) if era != "all" else pd.Series(True, index=dates.index)


def describe_cohort(frame: pd.DataFrame, era: str, cohort: str, protocol: dict) -> dict:
    a = frame.loc[era_mask(frame.date, era)]
    result = {"cohort": cohort, "era": era, "rows": len(a), "tickers": int(a.ticker.nunique()), "dates": int(a.date.nunique())}
    result["conditioning"] = "open_conditioned_idealized_open_mark" if "open_" in cohort or "recovery_" in cohort else "prior_close_signal_idealized_open_mark"
    result["secondary"] = cohort.startswith("secondary_")
    if a.empty:
        result.update({"sparse": True, "gross_mean_equal_date": None, "matched_excess_equal_date": None})
        return result
    daily = a.groupby("date").agg(gross=("return", "mean"), matched=("matched_residual", "mean"),
                                  baseline_all=("baseline_all", "mean"), baseline_excl=("baseline_excl", "mean"),
                                  n=("return", "size"), matched_n=("matched_residual", "count"))
    weeks = len(set(week_keys(daily.index)))
    gross = float(daily.gross.mean())
    matched = a.dropna(subset=["matched_residual"])
    lower, upper = a["return"].quantile([.01, .99])
    winsor = a.assign(clipped=a["return"].clip(lower, upper)).groupby("date").clipped.mean().mean()
    matched_winsor = None
    if not matched.empty:
        ml, mu = matched.matched_residual.quantile([.01, .99])
        matched_winsor = float(matched.assign(clipped=matched.matched_residual.clip(ml, mu)).groupby("date").clipped.mean().mean())
    max_day = daily.n.idxmax()
    finite_mean = lambda s: float(s.mean()) if s.notna().any() else None
    result.update({
        "active_weeks": weeks, "sparse": len(a) < 100 or len(daily) < 20 or weeks < 10,
        "gross_mean_equal_date": gross,
        "net_mean_equal_date_by_roundtrip_bps": {str(bps): gross - bps / 10000 for bps in protocol["cost_roundtrip_bps"]},
        "median_row_return": float(a["return"].median()),
        "positive_row_fraction_gross": float((a["return"] > 0).mean()),
        "positive_row_fraction_net25bps": float((a["return"] > .0025).mean()),
        "adverse_excursion_mean_row": float(a.adverse.mean()), "adverse_excursion_p05_row": float(a.adverse.quantile(.05)),
        "same_date_all_eligible_baseline_mean": finite_mean(daily.baseline_all),
        "same_date_nonmember_baseline_mean": finite_mean(daily.baseline_excl),
        "same_date_nonmember_excess_mean": finite_mean(daily.gross - daily.baseline_excl),
        "same_date_nonmember_covered_dates": int(daily.baseline_excl.notna().sum()),
        "matched_excess_equal_date": finite_mean(daily.matched),
        "matched_covered_gross_equal_date": float(matched.groupby("date")["return"].mean().mean()) if len(matched) else None,
        "matched_covered_control_equal_date": float(matched.groupby("date").matched_mean.mean().mean()) if len(matched) else None,
        "matched_control_covered_rows": len(matched), "matched_control_row_coverage": len(matched) / len(a),
        "matched_control_covered_dates": int(daily.matched.notna().sum()),
        "matched_control_partial_dates": int(((daily.matched_n > 0) & (daily.matched_n < daily.n)).sum()),
        "matched_excess_bootstrap": bootstrap_weekly(daily.matched, protocol["bootstrap"]["seed"], protocol["bootstrap"]["replications"]),
        "gross_winsor01_equal_date": float(winsor), "gross_winsor01_bounds": [float(lower), float(upper)],
        "matched_excess_winsor01_equal_date": matched_winsor,
        "abs_return_gt20pct_rows": int((a["return"].abs() > .2).sum()),
        "abs_return_gt50pct_rows": int((a["return"].abs() > .5).sum()),
        "abs_return_gt100pct_rows": int((a["return"].abs() > 1).sum()),
        "minimum_row_return": float(a["return"].min()), "maximum_row_return": float(a["return"].max()),
        "busiest_date": max_day.date().isoformat(), "busiest_date_rows": int(daily.loc[max_day, "n"]),
        "busiest_date_row_fraction": float(daily.loc[max_day, "n"] / len(a)),
        "top10_dates_row_fraction": float(daily.n.nlargest(10).sum() / len(a)),
    })
    return result


def mrcy_case(arrays: dict) -> dict:
    source = ROOT / "research/averaging_down_2026_11_06/source_data/daily_five_year/MRCY.json.gz"
    minute = ROOT / "research/averaging_down_2026_11_06/source_data/MRCY_2026-11-06_minute.json"
    a = arrays["MRCY"]
    dates = pd.DatetimeIndex(a["dates"])
    t = dates.get_loc("2026-11-06")
    bars = a["bars"]
    o, h, l, c, v = bars.T
    f = features(bars, valid_bars(bars)[0])
    prior = slice(t - 252, t)
    priorlow_i = t - 252 + int(np.argmin(l[prior]))
    doc = json.loads(minute.read_text())
    m = pd.DataFrame(doc["bars"])
    m.index = pd.to_datetime(m.t, unit="ms", utc=True).dt.tz_convert("America/New_York")
    first7 = m[(m.index >= "2026-11-06 09:30:00-05:00") & (m.index < "2026-11-06 09:37:00-05:00")]
    lowest = first7.l.idxmin()
    return {
        "role": "Separate retrospective motivating illustration; MRCY's full eligible history is included in the primary screen", "entry_date": "2026-11-06",
        "source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
        "minute_source": str(minute.relative_to(ROOT)), "minute_source_sha256": digest(minute),
        "previous_dates": [str(x.date()) for x in dates[t - 3:t]], "previous_closes": c[t - 3:t].tolist(),
        "prior_close_returns": [float(f["r2"][t]), float(f["r1"][t])], "two_day_close_return": float(c[t - 1] / c[t - 3] - 1),
        "prior252_low": float(f["prior_low"][t]), "prior252_low_date": str(dates[priorlow_i].date()),
        "prior_low_before_nov5": float(l[t - 253:t - 1].min()),
        "prior_low_before_nov5_date": str(dates[t - 253 + int(np.argmin(l[t - 253:t - 1]))].date()),
        "prior_close_near_low": bool(f["near_low"][t]), "prior_session_new252low": bool(f["new_low"][t]),
        "open": float(o[t]), "open_gap": float(o[t] / c[t - 1] - 1),
        "open_distance_above_prior252low": float(f["open_low_distance"][t]),
        "fraction_close_loss_recovered": float(f["recovery_numerator"][t] / f["recovery_denominator"][t]),
        "close": float(c[t]), "hypothetical_open_close_return": float(c[t] / o[t] - 1),
        "session_low": float(l[t]), "open_to_low_adverse_excursion": float(l[t] / o[t] - 1),
        "low_minute_start_et": lowest.isoformat(), "minute_low": float(first7.loc[lowest, "l"]),
        "first_seven_clock_minutes_observed_bars": len(first7),
        "timing_note": "Low occurs in minute starting09:36, six minutes after09:30; exact tick time unknown. Missing09:33 aggregate is not filled. Low is outcome-only, never morning input.",
        "actual_trade_note": "No November6 buys; remaining500 shares sold09:35–10:12. Actual campaign loss differs from hypothetical opening purchase. See existing mrcy_summary and latest_csv_reconciliation.",
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    protocol = json.loads((OUT / "protocol.json").read_text())
    snapshot = json.loads((OUT / "input_snapshot.json").read_text())
    assert protocol["protocol_version"] == 2
    assert protocol["status"] == "amended_after_initial_outcomes_user_requested_mrcy_inclusion"
    original_protocol = json.loads(original_artifact(protocol, "protocol.json"))
    for field in ["cohorts", "history_sessions", "minimum_open", "minimum_prior20_dollar_volume",
                  "shock_main", "shock_secondary", "shock_definition", "new_low", "near_low", "near_high",
                  "open_low_bins", "recovery_fraction", "recovery_bins", "eras", "target", "matching",
                  "cost_roundtrip_bps", "bootstrap", "winsorization", "source_cutoff"]:
        assert protocol[field] == original_protocol[field], f"Unrequested analytical change: {field}"
    source = ROOT / snapshot["arrays_source"]
    assert digest(source) == snapshot["arrays_sha256"], "immutable array hash changed"
    check_indexing()
    arrays = pd.read_pickle(source)
    registry = set(snapshot["registry_tickers"])
    held = set(snapshot["registry_holdout"])
    tickers = sorted(registry & arrays.keys() - held - {"SPY"})
    assert len(held) == 200 and len(registry) == 2000
    dates = pd.DatetimeIndex(arrays["SPY"]["dates"])
    assert dates.max().date().isoformat() == protocol["source_cutoff"]
    pieces = []
    quality = Counter()
    for ticker in tickers:
        a = arrays[ticker]
        assert list(a["dates"]) == list(arrays["SPY"]["dates"])
        bars = a["bars"]
        valid, counts = valid_bars(bars)
        quality.update(counts)
        quality["source_validity_mask_disagreements"] += int((valid != a["valid"]).sum())
        f = features(bars, valid)
        o, h, l, c, v = bars.T
        before_open = f["history_ok"] & (f["adv20"] >= protocol["minimum_prior20_dollar_volume"])
        eligible = before_open & np.isfinite(o) & (o >= protocol["minimum_open"])
        quality["rows_with_valid252_history"] += int(f["history_ok"].sum())
        quality["prior_eligible_missing_or_invalid_open"] += int((before_open & (~np.isfinite(o) | (o <= 0))).sum())
        quality["prior_eligible_positive_open_below5"] += int((before_open & np.isfinite(o) & (o > 0) & (o < 5)).sum())
        quality["entry_eligible_rows"] += int(eligible.sum())
        quality["entry_eligible_invalid_current_outcome"] += int((eligible & ~valid).sum())
        i = np.flatnonzero(eligible)
        # Keep all entry-eligible rows until after volatility strata are assigned.
        rows = {k: val[i] for k, val in f.items() if k not in {"history_ok", "adv20", "prior_low"}}
        rows.update({"date": dates[i], "ticker": ticker, "outcome_valid": valid[i],
                     "return": c[i] / o[i] - 1, "adverse": l[i] / o[i] - 1})
        pieces.append(pd.DataFrame(rows))
    panel = pd.concat(pieces, ignore_index=True)
    assert not panel.ticker.isin(held | {"SPY"}).any()
    assert "MRCY" in set(panel.ticker)
    assert panel.pre_shock_vol.notna().all()
    ranks = panel.groupby("date").pre_shock_vol.rank(method="first") - 1
    totals = panel.groupby("date").ticker.transform("size")
    panel["q"] = np.floor(ranks * 5 / totals).astype("int8")
    panel = panel.loc[panel.outcome_valid].copy()
    panel["ticker"] = panel.ticker.astype("category")
    assert panel["return"].notna().all() and (panel["return"] > -1).all()
    assert len(panel) == 1421271, "Amended discovery cohort count changed; investigate before proceeding"
    assert int((panel.ticker == "MRCY").sum()) == 995, "Full eligible MRCY history must be included"
    quality["evaluated_rows"] = len(panel)
    quality["evaluated_tickers"] = int(panel.ticker.nunique())
    quality["evaluated_dates"] = int(panel.date.nunique())
    quality["evaluated_mrcy_rows"] = int((panel.ticker == "MRCY").sum())
    quality["evaluated_mrcy_dates"] = int(panel.loc[panel.ticker == "MRCY", "date"].nunique())
    shock = (panel.r1 <= -.02) & (panel.r2 <= -.02)
    near = shock & panel.near_low
    recovery = panel.recovery_numerator / panel.recovery_denominator.where(shock)
    masks = {
        "all_eligible": np.ones(len(panel), dtype=bool), "shock_2pct": shock,
        "shock_2pct_new252low": shock & panel.new_low, "shock_2pct_close_near252low": near,
        "shock_2pct_nearlow_open_below_low": near & (panel.open_low_distance < 0),
        "shock_2pct_nearlow_open_0_to_1pct_above_low": near & panel.open_low_distance.between(0, .01),
        "shock_2pct_nearlow_open_gt1pct_above_low": near & (panel.open_low_distance > .01),
        "shock_2pct_recovery_negative": shock & (recovery < 0),
        "shock_2pct_recovery_0_to_25pct": shock & recovery.between(0, .25),
        "shock_2pct_recovery_gt25pct": shock & (recovery > .25),
        "all_eligible_close_near252high": panel.near_high,
    }
    for pct in [1, 3]:
        s = (panel.r1 <= -pct / 100) & (panel.r2 <= -pct / 100)
        masks[f"secondary_shock_{pct}pct"] = s
        masks[f"secondary_shock_{pct}pct_close_near252low"] = s & panel.near_low
    masks["all_eligible_close_near252low"] = panel.near_low
    masks["all_eligible_new252low"] = panel.new_low
    assert list(masks) == protocol["cohorts"]
    assert sum(int(masks[k].sum()) for k in ["shock_2pct_nearlow_open_below_low", "shock_2pct_nearlow_open_0_to_1pct_above_low", "shock_2pct_nearlow_open_gt1pct_above_low"]) == int(near.sum())
    assert sum(int(masks[k].sum()) for k in ["shock_2pct_recovery_negative", "shock_2pct_recovery_0_to_25pct", "shock_2pct_recovery_gt25pct"]) == int(shock.sum())
    stats = panel.groupby(["date", "q"])["return"].agg(["sum", "count"])
    daily_all = panel.groupby("date")["return"].agg(["sum", "count"])
    summaries, daily_records, mrcy_counts = [], [], {}
    for name, mask in masks.items():
        a = panel.loc[mask, ["date", "q", "ticker", "return", "adverse"]].copy()
        mrcy_counts[name] = int((a.ticker == "MRCY").sum())
        group = a.groupby(["date", "q"])["return"].agg(["sum", "count"])
        control = stats.subtract(group.reindex(stats.index, fill_value=0))
        control["matched_mean"] = (control["sum"] / control["count"]).where(control["count"] >= 5)
        a = a.join(control[["matched_mean"]], on=["date", "q"])
        a["matched_residual"] = a["return"] - a.matched_mean
        day = a.groupby("date")["return"].agg(["sum", "count"])
        dc = daily_all.subtract(day.reindex(daily_all.index, fill_value=0))
        baseline_excl = (dc["sum"] / dc["count"]).where(dc["count"] >= 5)
        a["baseline_all"] = a.date.map(daily_all["sum"] / daily_all["count"])
        a["baseline_excl"] = a.date.map(baseline_excl)
        for era in protocol["eras"]:
            summaries.append(describe_cohort(a, era, name, protocol))
        daily = a.groupby("date").agg(rows=("return", "size"), gross=("return", "mean"),
                                      matched_excess=("matched_residual", "mean"), matched_rows=("matched_residual", "count"),
                                      baseline_all=("baseline_all", "mean"), baseline_excl=("baseline_excl", "mean"))
        daily["cohort"] = name
        daily_records.extend(json.loads(daily.reset_index().to_json(orient="records", date_format="iso")))
        print(json.dumps({"cohort_complete": name, "rows": len(a), "dates": len(daily)}), flush=True)
    extreme = panel.loc[panel["return"].abs().nlargest(20).index, ["date", "ticker", "return", "adverse"]]
    write_json("results.json", {"protocol_sha256": digest(OUT / "protocol.json"), "unit": "fraction", "summaries": summaries})
    write_json("mrcy_inclusion_comparison.json", inclusion_comparison(protocol, summaries, mrcy_counts))
    write_json("daily_results.json", daily_records)
    write_json("data_quality.json", {"counts": dict(quality), "checks_passed": ["synthetic past-only feature indexing", "source hash", "excluded production holdout and SPY", "included all 995 eligible MRCY rows", "amended discovery row count", "disjoint recovery and open-low bin partitions"], "extreme_rows_retained": json.loads(extreme.to_json(orient="records", date_format="iso"))})
    write_json("mrcy_case.json", mrcy_case(arrays))
    write_json("run_manifest.json", {"started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
               "command": "bazel-bin/python/stock_picker/api/main.runfiles/rules_python++python+python_3_12_aarch64-apple-darwin/bin/python3 research/morning_dip_patterns_2026_11_09/run_screen.py",
               "script_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json"),
               "snapshot_sha256": digest(OUT / "input_snapshot.json"), "array_sha256": digest(source),
               "production_mutations": False, "holdout_tickers_evaluated": 0,
               "protocol_version": protocol["protocol_version"], "mrcy_included_in_primary": True,
               "mrcy_eligible_rows": quality["evaluated_mrcy_rows"],
               "original_results_commit": protocol["mrcy_inclusion_amendment"]["original_commit"],
               "method_verification": "Inspected signatures and installed source locations for pandas2.3.3 rolling,shift,rank,quantile,clip,GroupBy aggregate/mean/sum and NumPy2.5.2 RNG integers/percentile before calculation."})
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
