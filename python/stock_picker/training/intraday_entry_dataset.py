"""Bounded, reproducible intraday examples from frozen as-of Rank research.

No network requests or morning-model refits occur here. The historical cohort
uses one baseline score system, earlier-trained frozen score arrays, and cached
raw minute prices. An example's label is next-minute open to the same session's
final regular-minute close, with zero expense subtraction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_picker.features.intraday_entry import (
    FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, FEATURE_VERSION, MINUTE_MS,
    MORNING_FEATURE_COLUMNS, build_intraday_entry_features, normalize_rank_score,
)
from stock_picker.storage.paths import data_root

CHECKPOINTS_ET = ("09:35", "10:00", "10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "15:00")
ET = ZoneInfo("America/New_York")
TARGET_COLUMN = "target_return"


@dataclass(frozen=True)
class IntradayEntryDataset:
    frame: pd.DataFrame
    metadata: dict


def _read(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _identity(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame[["date", "ticker"]], index=False).to_numpy("uint64").tobytes()).hexdigest()


def _select_cached_baseline_rows(study: Path, repo: Path, cohort: list[dict], minute_keys: set[tuple[str, str]]):
    """Recover ranks 1–20 from the same frozen score system, without new fits.

    Cached lower ranks are an incomplete, selected sample: many were acquired
    because another research variant or saved live list chose them. The caller
    reports that limitation rather than representing this as full top-20 history.
    """
    experiment = repo / "research/morning_dip_model_2026_11_09"
    provenance = _read(study / "cohort_provenance.json")
    manifest_path = repo / provenance["manifest_path"]
    _require(_digest(manifest_path) == provenance["manifest_sha256"], "research manifest hash mismatch")
    manifest = _read(manifest_path)
    baseline = experiment / "baseline.parquet"
    _require(_digest(baseline) == provenance["baseline_sha256"], "frozen baseline hash mismatch")
    columns = ["date", "ticker", *MORNING_FEATURE_COLUMNS]
    panel = pd.read_parquet(baseline, columns=columns).sort_values(["date", "ticker"]).reset_index(drop=True)
    _require(not panel.duplicated(["date", "ticker"]).any(), "duplicate research row identities")
    baseline_rows = {r["session"]: r for r in cohort if r["cohort"] == "research_baseline"}
    _require(len(baseline_rows) == sum(r["cohort"] == "research_baseline" for r in cohort), "duplicate baseline dates")
    dates = np.sort(panel.date.unique())
    selected, score_sources = [], []
    for check in provenance["checkpoint_checks"]:
        if check["cohort"] != "research_baseline":
            continue
        array_path, metadata_path = repo / check["array_path"], repo / check["metadata_path"]
        _require(_digest(array_path) == check["array_sha256"], "baseline score array hash mismatch")
        _require(_digest(metadata_path) == check["metadata_sha256"], "baseline score metadata hash mismatch")
        metadata = _read(metadata_path)
        context = metadata["context"]
        test = panel.loc[panel.date.between(check["test_start"], check["test_end"])].copy()
        train_dates = dates[dates < np.datetime64(check["test_start"])][-manifest["protocol"]["training_sessions"]:]
        train = panel.loc[panel.date.isin(train_dates)]
        _require(_identity(test) == context["test_rows"] and _identity(train) == context["train_rows"], "score/date/ticker alignment mismatch")
        _require(train.date.max() < test.date.min() and str(train.date.max()) == context["fit_end"], "morning-score training chronology mismatch")
        scores = np.load(array_path, allow_pickle=False)
        _require(scores.shape == (len(test),) and np.isfinite(scores).all(), "invalid frozen score vector")
        test["rank_score"] = scores
        test = test.sort_values(["date", "rank_score", "ticker"], ascending=[True, False, True], kind="stable")
        test["rank"] = test.groupby("date").cumcount() + 1
        top = test.loc[test["rank"] <= 20]
        for day, daily in top.groupby("date", sort=False):
            session = day.date().isoformat()
            template = baseline_rows[session]
            first = daily.iloc[0]
            _require(first.ticker == template["ticker"] and math.isclose(first.rank_score, template["rank_score"], rel_tol=1e-12), "frozen Rank1 selection mismatch")
            top_scores = daily.rank_score.tolist()
            for record in daily.to_dict(orient="records"):
                if (record["ticker"], session) not in minute_keys:
                    continue
                selected.append({
                    "cohort": "research_baseline", "ticker": record["ticker"], "session": session,
                    "previous_session": template["previous_session"], "open_at": template["open_at"],
                    "close_at": template["close_at"], "training_end": check["training_end"],
                    "rank": int(record["rank"]), "rank_score": float(record["rank_score"]),
                    "rank_score_z": normalize_rank_score(record["rank_score"], top_scores),
                    "morning_features": {name: float(record[name]) for name in MORNING_FEATURE_COLUMNS},
                })
        score_sources.append({key: check[key] for key in ("block", "training_end", "array_path", "array_sha256", "metadata_path", "metadata_sha256")})
    _require(len({(r["ticker"], r["session"]) for r in selected}) == len(selected), "duplicate selected ticker-day")
    return selected, {"baseline_path": str(baseline.relative_to(repo)), "baseline_sha256": provenance["baseline_sha256"], "score_sources": score_sources}


def _daily_value(record: dict | None, ticker: str, session: str, field: str, repo: Path,
                 verified: dict[str, str] | None = None) -> float | None:
    # Legacy frozen daily snapshots predate the `complete` field. The successful
    # date/symbol-specific raw daily payload is their completeness contract.
    if not record or record.get("adjusted") is not False or record.get("complete") is False:
        return None
    source_path, expected = record.get("source_path"), record.get("source_sha256")
    if verified is None:
        verified = {}
    if source_path and source_path not in verified:
        verified[source_path] = _digest(repo / source_path)
    if not source_path or not expected or verified[source_path] != expected:
        raise ValueError("daily-source hash mismatch")
    payload = record.get("payload", {})
    valid_symbol = payload.get("symbol") == ticker or (
        record.get("provider_alias_verified") is True and payload.get("symbol") == record.get("provider_ticker"))
    if not valid_symbol or payload.get("status") != "OK" or payload.get("from") != session:
        return None
    value = payload.get(field)
    return float(value) if _positive(value) else None


def build_intraday_entry_dataset(
    study_dir: Path | str | None = None, *, checkpoints_et: Sequence[str] | None = None,
    expand_cached_top20: bool = True,
) -> IntradayEntryDataset:
    """Build all available fixed-clock states, including non-dips and rising prices.

    No sample is selected using its future return. Labels require an observed
    exact entry and final regular-minute close. Source problems remain explicit
    exclusions; they are never replaced by a zero-return/cash observation.
    ``expand_cached_top20=False`` permits small, isolated frozen-source fixtures.
    """
    repo = data_root().parent
    study = Path(study_dir) if study_dir is not None else repo / "research/rank_dip_timing_2026_11_09"
    checkpoints = tuple(checkpoints_et or CHECKPOINTS_ET)
    _require(len(checkpoints) == len(set(checkpoints)), "duplicate snapshot clocks")
    source_names = ("cohort.json", "minute_manifest.json", "daily_prices.json", "split_metadata.json")
    source_hashes = {name: _digest(study / name) for name in source_names}
    audit_path = study / "audit_checks.json"
    audit_verified = False
    if audit_path.exists():
        audit = _read(audit_path)
        _require(audit.get("status") == "passed", "frozen source audit did not pass")
        for name, observed_hash in source_hashes.items():
            _require(audit.get("artifact_sha256", {}).get(name) == observed_hash, f"frozen {name} hash mismatch")
        audit_verified = True
    else:
        _require(not expand_cached_top20, "frozen source audit is required for research cohort expansion")
    cohort = _read(study / "cohort.json")
    manifests = {(r["ticker"], r["session"]): r for r in _read(study / "minute_manifest.json")}
    daily_records = _read(study / "daily_prices.json")
    daily = {(r["ticker"], r.get("session", r.get("date"))): r for r in daily_records}
    _require(len(daily) == len(daily_records), "duplicate daily source identities")
    verified_daily = {}
    splits = _read(study / "split_metadata.json")
    if expand_cached_top20:
        selected, score_provenance = _select_cached_baseline_rows(study, repo, cohort, set(manifests))
    else:
        selected = [r for r in cohort if r["cohort"] == "research_baseline"]
        score_provenance = {"source": "provided frozen baseline cohort; score normalization may be unavailable"}
    _require(len({(r["ticker"], r["session"]) for r in selected}) == len(selected), "duplicate selected ticker-day")
    exclusions, examples, valid_stock_days = Counter(), [], set()
    for row in selected:
        ticker, session = row["ticker"], row["session"]
        _require(row["training_end"] < session, "morning model was not trained before this session")
        opened, closed = (datetime.fromisoformat(row[k]) for k in ("open_at", "close_at"))
        _require(opened.tzinfo is not None and closed.tzinfo is not None, "missing session timezone")
        source = manifests.get((ticker, session))
        if not source or source.get("complete") is not True or source.get("error_type"):
            exclusions["unavailable_minute_source"] += len(checkpoints)
            continue
        path = repo / source["path"]
        _require(_digest(path) == source["sha256"], "minute-source hash mismatch")
        document = _read(path)
        _require(document.get("ticker") == ticker and document.get("adjusted") is False, "minute-source identity or price basis mismatch")
        opening = _daily_value(daily.get((ticker, session)), ticker, session, "open", repo, verified_daily)
        bars = document.get("bars", [])
        by_time = {b["t"]: b for b in bars}
        _require(len(by_time) == len(bars), "duplicate raw minute timestamps")
        open_ms, close_ms = round(opened.timestamp() * 1000), round(closed.timestamp() * 1000)
        initial, exit_bar = by_time.get(open_ms), by_time.get(close_ms - MINUTE_MS)
        issue = "missing_official_open" if opening is None else "missing_session_first_minute" if not initial else "opening_price_basis_conflict" if not _positive(initial.get("o")) or abs(initial["o"] / opening - 1) > .01 else "missing_final_exit_minute" if not exit_bar or not _positive(exit_bar.get("c")) else None
        if issue:
            exclusions[issue] += len(checkpoints)
            continue
        previous = _daily_value(daily.get((ticker, row["previous_session"])), ticker, row["previous_session"], "close", repo, verified_daily)
        split_coverage = (splits.get("complete") is True and splits.get("coverage_start", "9999") <= row["previous_session"]
                          and splits.get("coverage_end", "") >= session)
        provider_ticker = daily.get((ticker, session), {}).get("provider_ticker", ticker)
        crosses_split = any(e.get("ticker") in (ticker, provider_ticker) and row["previous_session"] < e["execution_date"] <= session for e in splits.get("relevant_events", []))
        if not split_coverage or crosses_split:
            previous = None
        for clock in checkpoints:
            hour, minute = map(int, clock.split(":"))
            observed = opened.astimezone(ET).replace(hour=hour, minute=minute, second=0, microsecond=0)
            if not opened < observed < closed or observed.hour * 60 + observed.minute < 575:
                exclusions["outside_eligible_session_time"] += 1
                continue
            entry_bar = by_time.get(round(observed.timestamp() * 1000))
            if not entry_bar or not _positive(entry_bar.get("o")):
                exclusions["missing_exact_entry_minute"] += 1
                continue
            entry = float(entry_bar["o"])
            try:
                features = build_intraday_entry_features(
                    bars, observed_at=observed, session_open_at=opened, session_close_at=closed,
                    entry_price=entry, session_open=opening, previous_close=previous,
                    rank=row["rank"], rank_score=row["rank_score"], rank_score_z=row.get("rank_score_z"),
                    morning_features=row.get("morning_features"),
                )
            except ValueError as exc:
                exclusions[str(exc)] += 1
                continue
            label = float(exit_bar["c"]) / entry - 1
            _require(math.isfinite(label), "nonfinite entry-to-close label")
            examples.append({"session": session, "ticker": ticker, "rank": row["rank"],
                             "observed_at": observed.isoformat(), "entry_price": entry,
                             "label_observed_at": closed.isoformat(), "morning_training_end": row["training_end"],
                             "target_return": label, **features})
            valid_stock_days.add((ticker, session))
    frame = pd.DataFrame(examples, columns=["session", "ticker", "rank", "observed_at", "entry_price", "label_observed_at", "morning_training_end", TARGET_COLUMN, *FEATURE_COLUMNS])
    if not frame.empty:
        frame = frame.sort_values(["session", "ticker", "observed_at"]).reset_index(drop=True)
    metadata = {
        "feature_version": FEATURE_VERSION, "feature_columns": list(FEATURE_COLUMNS),
        "feature_descriptions": FEATURE_DESCRIPTIONS, "expense_fraction": 0.0,
        "target": "Final regular-minute close / exact entry-minute open - 1; zero expense",
        "availability_assumption_et": "09:35", "checkpoints_et": list(checkpoints),
        "cohort": "Frozen baseline morning Rank, cached top-20 candidates; never pooled with all-21 or mutable live scores",
        "candidate_stock_days": len(selected), "candidate_snapshots": len(selected) * len(checkpoints),
        "rows": len(frame), "unique_stock_days": len(valid_stock_days),
        "unique_sessions": int(frame.session.nunique()) if not frame.empty else 0,
        "unique_tickers": int(frame.ticker.nunique()) if not frame.empty else 0,
        "first_session": frame.session.min() if not frame.empty else None,
        "last_session": frame.session.max() if not frame.empty else None,
        "candidate_rank_counts": dict(sorted(Counter(r["rank"] for r in selected).items())),
        "rank_snapshot_counts": {int(k): int(v) for k, v in frame["rank"].value_counts().sort_index().items()} if not frame.empty else {},
        "excluded_snapshots": dict(sorted(exclusions.items())),
        "missing_feature_counts": {column: int(frame[column].isna().sum()) for column in FEATURE_COLUMNS},
        "source_sha256": source_hashes, "frozen_audit_verified": audit_verified,
        "daily_source_records": len(daily_records), "verified_daily_source_files_used": len(verified_daily),
        "legacy_daily_source_records": sum("complete" not in r for r in daily_records),
        "score_provenance": score_provenance,
        "morning_feature_contract": "Six close-known columns from the frozen baseline training frame, already shifted by build_training_frame; never read the current day's live feature row.",
        "limitations": [
            "Earlier-trained research baseline rankings, not immutable historical live publications or the current production feature subset.",
            "Lower ranks are selectively cached, with sparse support; this is not complete top-20 coverage.",
            "Repeated intraday snapshots are correlated; split and weight by whole session dates.",
            "Minute bars are hypothetical execution prices; missing observations remain unknown.",
            "Top-20 score normalization removes arbitrary scale but does not establish transfer calibration across morning-model versions.",
        ],
    }
    _require(len(frame) + sum(exclusions.values()) == len(selected) * len(checkpoints), "snapshot counts do not reconcile")
    return IntradayEntryDataset(frame=frame, metadata=metadata)
