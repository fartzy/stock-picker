"""Isolated, frozen 151-versus-177 input Rank comparison; never publishes a model.

Uses the production Rank trainer and chronological SVM stack without changing
their parameters. Large snapshots, OOF rows and model archives stay in .runtime.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from stock_picker.features.kronos import CONTRACT_ATTR, KRONOS_COLUMNS, historical_inputs, ready_contract
from stock_picker.features.pattern_evidence import PATTERN_EVIDENCE_COLUMNS, build_pattern_evidence
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.ingestion.kronos import normalized_history, retained_sessions
from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.kronos_store import KronosStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.paths import data_root
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.dataset import LABEL_COLUMN, build_training_frame
from stock_picker.training.dip_feature_experiment import paired_week_interval
from stock_picker.training.ensemble import Ensemble, ModelSpec, predict_ensemble, train_ensemble
from stock_picker.training.feature_revisit import daily_results, save_json, sha256
from stock_picker.training.model import DEFAULT_NUM_BOOST_ROUND, LIGHTGBM_DEFAULT_PARAMS, feature_columns, fit_stacked_svm_estimators, predict
from stock_picker.training.splits import select_holdout_tickers
from stock_picker.training.svm_stack import iter_stacked_svm_folds

EXPERIMENT = "rank_new_inputs_2026_11_12"
THROUGH = "2026-11-11"
N_SPLITS = 4
METRICS = ("top1", "top5", "top20", "hit1", "hit5", "hit20", "rank_ic")
NEW_COLUMNS = (*PATTERN_EVIDENCE_COLUMNS, *KRONOS_COLUMNS, *STACKED_SVM_COLUMNS)


def progress(runtime: Path, event: str, **details) -> None:
    state = {"updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "event": event, **details}
    save_json(runtime / "state.json", state)
    print(json.dumps(state, allow_nan=False), flush=True)


def source_hashes(root: Path) -> dict[str, str]:
    """Pin upstream builders/trainers too, not only this research entry point."""
    sources = {str(path.relative_to(root)): sha256(path)
            for family in ("features", "training", "ingestion", "storage")
            for path in sorted((root / "python/stock_picker" / family).glob("*.py"))}
    sources["tools/kronos/contract.py"] = sha256(root / "tools/kronos/contract.py")
    return sources


def validate_features(raw: list[str], excluded: set[str]) -> tuple[list[str], list[str]]:
    baseline = sorted(set(raw) - excluded)
    if (len(raw), len(excluded), len(baseline)) != (246, 95, 151):
        raise ValueError("expected frozen 246 raw / 95 excluded / 151 active inputs")
    if not {"dip_recovery_fraction", "dip_volume_change_1d"} <= excluded:
        raise ValueError("the two defective dip ratios must remain excluded")
    if set(raw) & set(NEW_COLUMNS) or excluded & set(NEW_COLUMNS):
        raise ValueError("new inputs already present or excluded; review the frozen comparison")
    return baseline, sorted(set(baseline) | set(NEW_COLUMNS))


def prepare(runtime: Path, through: str) -> dict:
    """Read sources once; never write back to prices/features/config/registry."""
    root = data_root().parent
    sources = source_hashes(root)
    tickers = sorted(UniverseStore().active_tickers())
    holdout = select_holdout_tickers(tickers)
    if (len(tickers), len(holdout)) != (2000, 200):
        raise ValueError("expected the authorized 2000-stock universe and 200-stock holdout")
    pruned = PrunedFeatureStore().read_all()
    excluded = {entry["feature"] for entry in pruned}
    contract = KronosStore().collection_contract()
    if contract is None:
        raise ValueError("no pinned Kronos collection contract")
    prices, features = PriceStore(), FeatureStore()
    snapshot = runtime / "histories"
    snapshot.mkdir(parents=True, exist_ok=True)
    frames, inputs, missing, snapshots = [], [], [], {}
    sessions = retained_sessions(through)
    for position, ticker in enumerate(tickers, 1):
        paths = [data_root() / family / f"{ticker}.parquet" for family in ("prices", "features")]
        if not all(path.exists() for path in paths):
            missing.append(ticker)
            continue
        before = [sha256(path) for path in paths]
        history = normalized_history(prices.read(ticker)).loc[:through]
        stored = normalized_history(features.read(ticker)).loc[:through]
        frame = build_training_frame(history, stored)
        # Evidence is already decision-day aligned: do not shift it a second time.
        frame = frame.join(build_pattern_evidence(history).reindex(frame.index))
        frame = frame.loc[frame.index.isin(pd.to_datetime(sessions))]
        if before != [sha256(path) for path in paths]:
            raise RuntimeError(f"source changed during read: {ticker}")
        inputs.extend({"path": str(path.relative_to(root)), "sha256": digest}
                      for path, digest in zip(paths, before))
        history_path = snapshot / f"{ticker}.parquet"
        history.to_parquet(history_path)
        snapshots[str(history_path.relative_to(runtime))] = sha256(history_path)
        frames.append(frame.reset_index(names="date").assign(ticker=ticker))
        if position % 100 == 0:
            progress(runtime, "freezing_sources", stocks=position, total=len(tickers))
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    if panel.empty or panel.duplicated(["date", "ticker"]).any():
        raise ValueError("empty or duplicate stock-day panel")
    raw = sorted(set(feature_columns(panel)) - set(PATTERN_EVIDENCE_COLUMNS))
    baseline, candidate = validate_features(raw, excluded)
    numeric = feature_columns(panel)
    infinite = int(np.isinf(panel[numeric].to_numpy(dtype=float)).sum())
    panel[numeric] = panel[numeric].replace([np.inf, -np.inf], np.nan)
    panel.to_parquet(runtime / "raw_panel.parquet", index=False)
    snapshots["raw_panel.parquet"] = sha256(runtime / "raw_panel.parquet")
    if sources != source_hashes(root):
        raise RuntimeError("research source changed during freeze")
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(), "through": through,
        "tickers": tickers, "holdout_tickers": sorted(holdout), "prune_records": pruned,
        "excluded": sorted(excluded), "baseline": baseline, "candidate": candidate,
        "svm_source_features": sorted(set(candidate) - set(STACKED_SVM_COLUMNS)),
        "source_hashes": sources, "source_inputs": inputs, "snapshots": snapshots,
        "kronos_contract": contract, "sessions": sessions, "missing_sources": missing,
        "rows": len(panel), "panel_sessions": int(panel.date.nunique()),
        "panel_start": str(panel.date.min().date()), "panel_end": str(panel.date.max().date()),
        "infinities_marked_missing": infinite, "n_splits": N_SPLITS,
        "rank_params": LIGHTGBM_DEFAULT_PARAMS, "rank_rounds": DEFAULT_NUM_BOOST_ROUND,
        "packages": {name: version(name) for name in ("lightgbm", "numpy", "pandas", "scikit-learn", "pyarrow")},
        "assumed_costs_bps": 0, "special_ticker_exclusions": [],
        "primary_metric": "daily top1 gross open-to-close return",
        "uncertainty": "paired ISO-week cluster bootstrap, 2000 draws, seed 42; exploratory",
        "interpretation": "combined candidate comparison only; gains do not establish individual feature lift",
    }
    save_json(runtime / "protocol.json", protocol)
    progress(runtime, "sources_frozen", rows=len(panel), sessions=protocol["panel_sessions"])
    return protocol


def checked_protocol(runtime: Path) -> dict:
    protocol = json.loads((runtime / "protocol.json").read_text())
    if protocol["source_hashes"] != source_hashes(data_root().parent):
        raise ValueError("code changed after source freeze; use a new experiment directory")
    for relative, expected in protocol["snapshots"].items():
        if sha256(runtime / relative) != expected:
            raise ValueError(f"frozen snapshot hash mismatch: {relative}")
    if KronosStore().collection_contract() != protocol["kronos_contract"]:
        raise ValueError("Kronos collection contract changed after freeze")
    return protocol


def completed_prefix(store: KronosStore, protocol: dict) -> list[str]:
    """A hole stops the provisional period, even if later dates are complete."""
    prefix = []
    for session in protocol["sessions"]:
        if not store.coverage(protocol["kronos_contract"], protocol["tickers"], [session])["ready"]:
            break
        prefix.append(session)
    if len(prefix) < (N_SPLITS + 1) * 4:
        raise ValueError("pilot needs at least 20 contiguous completed sessions")
    return prefix


def materialize(runtime: Path, protocol: dict, sessions: list[str]) -> pd.DataFrame:
    panel = pd.read_parquet(runtime / "raw_panel.parquet")
    panel = panel[panel.date.isin(pd.to_datetime(sessions))].copy()
    for name in KRONOS_COLUMNS:
        panel[name] = np.nan
    store = KronosStore()
    for position, (ticker, rows) in enumerate(panel.groupby("ticker", sort=True), 1):
        history = pd.read_parquet(runtime / "histories" / f"{ticker}.parquet")
        inputs = historical_inputs(ticker, history, store, protocol["kronos_contract"])
        panel.loc[rows.index, list(KRONOS_COLUMNS)] = inputs.reindex(pd.DatetimeIndex(rows.date)).to_numpy()
        if position % 100 == 0:
            progress(runtime, "joining_kronos", stocks=position, through=sessions[-1])
    panel.attrs[CONTRACT_ATTR] = protocol["kronos_contract"]
    return panel


def summary(daily: pd.DataFrame) -> dict:
    means = {metric: (float(daily[metric].mean()) if daily[metric].notna().any() else None) for metric in METRICS}
    return {**means, "days": len(daily), "rows": int(daily.stocks.sum()),
            "start": daily.date.min(), "end": daily.date.max()}


def comparison(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    return {"baseline": summary(baseline), "candidate": summary(candidate),
            "paired_candidate_minus_baseline": {
                metric: paired_week_interval(candidate.to_dict("records"), baseline.to_dict("records"), metric, 2000, 42)
                for metric in METRICS}}


def pair(ensemble: Ensemble, estimators: dict) -> None:
    ensemble.stacked_svm_estimators = estimators
    ensemble.direction_svc = estimators["svc_direction_margin"]


def validate_estimators(estimators: dict, protocol: dict) -> None:
    if set(estimators) != set(STACKED_SVM_COLUMNS) or any(
        set(model.feature_names) != set(protocol["svm_source_features"])
        for model in estimators.values()
    ):
        raise ValueError("all seven SVMs must consume exactly the frozen 170 raw inputs")


def evaluate(frame: pd.DataFrame, models: dict[str, Ensemble], *, stacked: bool) -> tuple[dict, pd.DataFrame]:
    results, scores = {}, frame[["date", "ticker", LABEL_COLUMN]].copy()
    for name, model in models.items():
        # Fold test rows already contain the exact paired predictions. Never
        # re-score OOF training rows with full-period source estimators.
        values = predict(model.members[0], frame) if stacked else predict_ensemble(model, frame)
        results[name] = daily_results(frame, values).drop(columns="mae")
        scores[name] = values
    return results, scores


def run_stage(runtime: Path, reports: Path, protocol: dict, stage: str) -> dict:
    destination = runtime / stage
    if (destination / "complete.json").exists():
        completed = json.loads((destination / "complete.json").read_text())
        if completed["protocol_sha256"] != sha256(runtime / "protocol.json"):
            raise ValueError("completed stage protocol hash mismatch")
        for path, expected in completed["artifact_hashes"].items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"completed stage artifact hash mismatch: {path}")
        return completed
    if destination.exists():
        raise FileExistsError(f"partial stage exists: {destination}; inspect failure before starting a new experiment")
    store = KronosStore()
    if stage == "full":
        contract, coverage = ready_contract(store, protocol["tickers"], protocol["through"])
        if contract != protocol["kronos_contract"]:
            raise RuntimeError(f"full Kronos readiness required: {coverage}")
        sessions = protocol["sessions"]
    else:
        sessions = completed_prefix(store, protocol)
        coverage = store.coverage(protocol["kronos_contract"], protocol["tickers"], sessions)
    destination.mkdir()
    save_json(destination / "period.json", {"stage": stage, "sessions": sessions, "coverage": coverage})
    panel = materialize(runtime, protocol, sessions)
    valid = np.isfinite(panel[list(KRONOS_COLUMNS)]).all(axis=1)
    coverage_rows = panel[["date", "ticker"]].assign(valid=valid)
    for key in ("date", "ticker"):
        grouped = coverage_rows.groupby(key).valid.agg(["sum", "count"])
        grouped.columns = ["finite_kronos_rows", "eligible_rows"]
        grouped["finite_fraction"] = grouped.finite_kronos_rows / grouped.eligible_rows
        grouped.to_csv(reports / f"{stage}_kronos_by_{key}.csv")
    if stage == "full" and not coverage_rows.groupby("date").valid.any().all():
        raise ValueError("full stage has a session without any exact-source valid Kronos inputs")
    panel.to_parquet(destination / "panel.parquet")
    training = panel[~panel.ticker.isin(protocol["holdout_tickers"])]
    holdout = panel[panel.ticker.isin(protocol["holdout_tickers"])]
    if set(feature_columns(training, set(protocol["excluded"]))) != set(protocol["svm_source_features"]):
        raise ValueError("SVM source schema differs from frozen inputs")
    specs = {name: ModelSpec("lightgbm_rank", excluded_features=set(protocol["excluded"]),
                            included_features=set(protocol[name])) for name in ("baseline", "candidate")}
    archives = ModelStore(destination / "models")
    daily = {cohort: {name: [] for name in specs} for cohort in ("training_tickers", "holdout_tickers")}
    folds, gains = [], []
    iterator = iter_stacked_svm_folds(training, outputs=STACKED_SVM_COLUMNS,
                                    excluded_features=set(protocol["excluded"]), n_splits=N_SPLITS)
    for index in range(N_SPLITS):
        progress(runtime, "fitting_fold_svms", stage=stage, fold=index + 1, folds=N_SPLITS)
        fold = next(iterator)
        validate_estimators(fold.estimators, protocol)
        if not fold.baseline_train.index.equals(fold.stacked_train.index):
            raise ValueError("baseline/candidate fitting rows differ")
        if fold.stacked_train.date.max() >= fold.stacked_test.date.min():
            raise ValueError("training dates overlap evaluation dates")
        models = {"baseline": train_ensemble(fold.baseline_train, [specs["baseline"]]),
                  "candidate": train_ensemble(fold.stacked_train, [specs["candidate"]])}
        pair(models["candidate"], fold.estimators)
        fold_record = {"fold": index + 1, "train_rows": len(fold.stacked_train),
                       "train_start": str(fold.stacked_train.date.min().date()),
                       "train_end": str(fold.stacked_train.date.max().date())}
        for cohort, evaluation, stacked in (
            ("training_tickers", fold.stacked_test, True),
            ("holdout_tickers", holdout[holdout.date.isin(fold.stacked_test.date.unique())], False),
        ):
            if evaluation.empty:
                raise ValueError(f"no {cohort} evaluation rows")
            results, scores = evaluate(evaluation, models, stacked=stacked)
            scores.to_parquet(destination / f"fold_{index + 1}_{cohort}_scores.parquet", index=False)
            for name in specs:
                daily[cohort][name].append(results[name])
            fold_record[cohort] = comparison(results["baseline"], results["candidate"])
        for name, model in models.items():
            archives.write(f"fold_{index + 1}_{name}", model)
        member = models["candidate"].members[0]
        gain = dict(zip(member.feature_names, map(float, member.estimator.feature_importance("gain"))))
        gains.append({name: gain[name] for name in NEW_COLUMNS})
        folds.append(fold_record)
        save_json(destination / "folds.json", {"folds": folds, "new_feature_gains": gains})
        progress(runtime, "fold_complete", stage=stage, fold=index + 1)
    # Exactly the same OOF refit protocol as training/train.py:231-249.
    final = pd.concat([fold.stacked_train, fold.stacked_test])
    final.attrs[CONTRACT_ATTR] = protocol["kronos_contract"]
    if not final.index.is_unique:
        raise ValueError("final OOF rows must be unique")
    final.to_parquet(destination / "final_oof.parquet")
    progress(runtime, "fitting_final_svms", stage=stage, rows=len(training))
    final_estimators = fit_stacked_svm_estimators(training,
        excluded_features=set(protocol["excluded"]) | set(STACKED_SVM_COLUMNS), outputs=STACKED_SVM_COLUMNS)
    validate_estimators(final_estimators, protocol)
    for name in specs:
        model = train_ensemble(final, [specs[name]])
        if name == "candidate":
            pair(model, final_estimators)
        archives.write(f"final_{name}", model)
    result = {"stage": stage, "provisional": stage == "pilot", "through": sessions[-1],
              "assumed_costs_bps": 0, "coverage": coverage, "folds": folds,
              "new_feature_gains_by_fold": gains,
              "kronos_finite_rows": int(np.isfinite(panel[list(KRONOS_COLUMNS)]).all(axis=1).sum()),
              "panel_rows": len(panel), "panel_sessions": int(panel.date.nunique()),
              "panel_start": str(panel.date.min().date()), "panel_end": str(panel.date.max().date()),
              "retained_sessions_without_eligible_rows": sorted(set(sessions) - set(panel.date.dt.strftime("%Y-%m-%d"))),
              "panel_sha256": sha256(destination / "panel.parquet"),
              "protocol_sha256": sha256(runtime / "protocol.json"),
              "final_head_rows": len(final), "final_svm_rows": len(training),
              "status": "retrospective isolated research; no live model change; no family ablation claim"}
    for cohort, names in daily.items():
        joined = {name: pd.concat(blocks, ignore_index=True) for name, blocks in names.items()}
        for name, values in joined.items():
            values.to_csv(reports / f"{stage}_{cohort}_{name}_daily.csv", index=False)
        result[cohort] = comparison(joined["baseline"], joined["candidate"])
    save_json(reports / f"{stage}_results.json", result)
    result["artifact_hashes"] = {str(path.resolve()): sha256(path) for path in (
        *sorted(destination.rglob("*.parquet")), *sorted((destination / "models").glob("*.pkl")),
        *sorted(reports.glob(f"{stage}_*.csv")), reports / f"{stage}_results.json")}
    save_json(destination / "complete.json", result)
    progress(runtime, "stage_complete", stage=stage)
    return result


def check_backfill_process(process: dict) -> None:
    """A reused or zombie PID must not keep an unattended experiment waiting."""
    started = datetime.fromisoformat(process["started_at"])
    command = process["command"]
    limit = float(command[command.index("--max-seconds") + 1])
    if datetime.now(timezone.utc) > started + timedelta(seconds=limit + 300):
        raise RuntimeError("Kronos backfill deadline exceeded (configured limit plus five-minute grace)")
    observed = subprocess.run(
        ["ps", "-p", str(process["pid"]), "-o", "lstart=", "-o", "stat=", "-o", "command="],
        capture_output=True, text=True, check=False, timeout=10, env={**os.environ, "LC_ALL": "C"},
    )
    fields = observed.stdout.strip().split(maxsplit=6)
    if observed.returncode or len(fields) != 7 or "Z" in fields[5]:
        raise RuntimeError("Kronos backfill stopped before full readiness")
    actual_start = datetime.strptime(" ".join(fields[:5]), "%a %b %d %H:%M:%S %Y").astimezone()
    # caffeinate execs the Bazel launcher, which in turn execs its Python
    # stage2 bootstrap. The original executable prefix and all CLI arguments
    # survive, but the caffeinate/-i launcher tokens do not.
    expected = command[2:] if Path(command[0]).name == "caffeinate" and command[1] == "-i" else command
    if abs((actual_start - started).total_seconds()) > 10 or not all(part in fields[6] for part in expected):
        raise RuntimeError("Kronos backfill process identity changed; refusing stale/reused PID")


def wait_for_full(runtime: Path, protocol: dict, process_path: Path, poll_seconds: float) -> None:
    while True:
        contract, coverage = ready_contract(KronosStore(), protocol["tickers"], protocol["through"])
        if contract == protocol["kronos_contract"]:
            return
        if KronosStore().collection_contract() != protocol["kronos_contract"]:
            raise RuntimeError("Kronos collection contract changed while waiting")
        if any(coverage.get("statuses", {}).get(status, 0) for status in ("worker_error", "invalid_request")):
            raise RuntimeError(f"Kronos backfill has unresolved failures: {coverage}")
        process = json.loads(process_path.read_text())
        check_backfill_process(process)
        progress(runtime, "waiting_for_full_kronos", coverage=coverage, backfill_pid=process["pid"])
        time.sleep(poll_seconds)


def main() -> None:
    root = data_root().parent.resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "pilot", "full", "pilot-then-wait-full"), required=True)
    parser.add_argument("--runtime", type=Path, default=root / ".runtime/research" / EXPERIMENT)
    parser.add_argument("--reports", type=Path, default=root / "research" / EXPERIMENT)
    parser.add_argument("--through", default=THROUGH)
    parser.add_argument("--backfill-process", type=Path, default=root / ".runtime/kronos/backfill-process.json")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    runtime, reports = args.runtime.resolve(), args.reports.resolve()
    if not runtime.is_relative_to(root / ".runtime/research") or not reports.is_relative_to(root / "research"):
        parser.error("runtime must be under .runtime/research and reports under research")
    if args.through != THROUGH or not 1 <= args.poll_seconds <= 60:
        parser.error(f"this frozen experiment requires --through {THROUGH} and polling between 1 and 60 seconds")
    runtime.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    with (runtime / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not (runtime / "protocol.json").exists():
                prepare(runtime, args.through)
            protocol = checked_protocol(runtime)
            frozen_report = reports / "frozen_protocol.json"
            if frozen_report.exists() and json.loads(frozen_report.read_text())["runtime_protocol_sha256"] != sha256(runtime / "protocol.json"):
                raise ValueError("report directory belongs to another frozen run; choose fresh --runtime AND --reports")
            save_json(frozen_report, {
                key: value for key, value in protocol.items() if key not in {"source_inputs", "snapshots"}
            } | {"runtime_protocol_sha256": sha256(runtime / "protocol.json"),
                 "source_input_count": len(protocol["source_inputs"])})
            if args.mode == "prepare":
                return
            if args.mode in ("pilot", "pilot-then-wait-full"):
                run_stage(runtime, reports, protocol, "pilot")
            if args.mode == "pilot-then-wait-full":
                wait_for_full(runtime, protocol, args.backfill_process, args.poll_seconds)
                protocol = checked_protocol(runtime)
            if args.mode in ("full", "pilot-then-wait-full"):
                run_stage(runtime, reports, protocol, "full")
        except Exception as exc:
            progress(runtime, "failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
