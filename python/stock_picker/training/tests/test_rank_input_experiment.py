"""Safety boundaries of the isolated research orchestration, without expensive fits."""

from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import sys

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.kronos import CONTRACT_ATTR, KRONOS_COLUMNS
from stock_picker.features.pattern_evidence import PATTERN_EVIDENCE_COLUMNS
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training import rank_input_experiment as experiment
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.svm_stack import StackedSvmFold


def test_frozen_counts_keep_all_prunes_including_defective_ratios():
    excluded = {"dip_recovery_fraction", "dip_volume_change_1d", *[f"pruned_{i}" for i in range(93)]}
    raw = [*[f"active_{i}" for i in range(151)], *sorted(excluded)]
    baseline, candidate = experiment.validate_features(raw, excluded)
    assert len(baseline) == 151
    assert len(candidate) == 177
    assert set(candidate) - set(baseline) == set(experiment.NEW_COLUMNS)
    assert not set(candidate) & excluded
    with pytest.raises(ValueError, match="246 raw"):
        experiment.validate_features(raw[:-1], excluded)
    with pytest.raises(ValueError, match="defective dip"):
        experiment.validate_features(["new_prune" if x == "dip_recovery_fraction" else x for x in raw],
                                     excluded - {"dip_recovery_fraction"} | {"new_prune"})


def test_completed_prefix_never_bridges_a_hole(monkeypatch):
    monkeypatch.setattr(experiment, "N_SPLITS", 1)
    sessions = [str(day.date()) for day in pd.bdate_range("2026-01-01", periods=12)]
    seen = []

    def coverage(contract, tickers, dates):
        seen.append(dates[0])
        return {"ready": dates[0] != sessions[9]}

    store = SimpleNamespace(coverage=coverage)
    protocol = {"sessions": sessions, "tickers": ["A"], "kronos_contract": {}}
    assert experiment.completed_prefix(store, protocol) == sessions[:9]
    assert seen == sessions[:10]


def test_prepare_freezes_sources_and_joins_patterns_on_decision_day(monkeypatch, tmp_path):
    root = tmp_path / "data"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for family in ("prices", "features"):
        (root / family).mkdir(parents=True)
        (root / family / "MRCY.parquet").write_bytes(b"frozen source")
    excluded = {"dip_recovery_fraction", "dip_volume_change_1d", *[f"pruned_{i}" for i in range(93)]}
    dates = pd.bdate_range("2026-11-09", periods=4)
    history = pd.DataFrame({"Open": [10.] * 4, "High": [12.] * 4, "Low": [9.] * 4,
                            "Close": [11.] * 4, "Volume": [1000.] * 4}, index=dates)
    stored = pd.DataFrame(1., index=dates, columns=[*[f"active_{i}" for i in range(151)], *sorted(excluded)])
    tickers = ["MRCY", *[f"T{i}" for i in range(1999)]]
    monkeypatch.setattr(experiment, "data_root", lambda: root)
    monkeypatch.setattr(experiment, "source_hashes", lambda root: {})
    monkeypatch.setattr(experiment, "version", lambda name: "test")
    monkeypatch.setattr(experiment, "UniverseStore", lambda: SimpleNamespace(active_tickers=lambda: tickers))
    monkeypatch.setattr(experiment, "PriceStore", lambda: SimpleNamespace(read=lambda ticker: history))
    monkeypatch.setattr(experiment, "FeatureStore", lambda: SimpleNamespace(read=lambda ticker: stored))
    monkeypatch.setattr(experiment, "PrunedFeatureStore", lambda: SimpleNamespace(read_all=lambda: [{"feature": name} for name in excluded]))
    monkeypatch.setattr(experiment, "KronosStore", lambda: SimpleNamespace(collection_contract=lambda: {"frozen": True}))

    def patterns(frame):
        return pd.DataFrame({name: frame.index.day for name in PATTERN_EVIDENCE_COLUMNS}, index=frame.index)

    monkeypatch.setattr(experiment, "build_pattern_evidence", patterns)
    protocol = experiment.prepare(runtime, "2026-11-11")
    frozen = pd.read_parquet(runtime / "raw_panel.parquet")
    assert frozen.date.dt.day.tolist() == [10, 11]
    assert frozen[PATTERN_EVIDENCE_COLUMNS[0]].tolist() == [10, 11]
    assert len(protocol["baseline"]) == 151
    assert len(protocol["candidate"]) == 177
    assert len(protocol["holdout_tickers"]) == 200
    assert protocol["special_ticker_exclusions"] == []
    assert frozen.ticker.unique().tolist() == ["MRCY"]
    assert pd.read_parquet(runtime / "histories/MRCY.parquet").index.max() == pd.Timestamp("2026-11-11")
    assert len(protocol["source_inputs"]) == 2


def test_checked_snapshot_rejects_mutation(monkeypatch, tmp_path):
    frozen = tmp_path / "raw_panel.parquet"
    frozen.write_bytes(b"original")
    experiment.save_json(tmp_path / "protocol.json", {"source_hashes": {}, "snapshots": {
        "raw_panel.parquet": experiment.sha256(frozen)}})
    monkeypatch.setattr(experiment, "source_hashes", lambda root: {})
    frozen.write_bytes(b"modified")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        experiment.checked_protocol(tmp_path)


def test_full_readiness_fails_before_materialization_or_training(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "ready_contract", lambda *args: (None, {"ready": False}))
    monkeypatch.setattr(experiment, "materialize", lambda *args: pytest.fail("must not materialize"))
    protocol = {"tickers": ["A"], "through": "2026-11-11", "kronos_contract": {"pinned": True}}
    with pytest.raises(RuntimeError, match="full Kronos readiness required"):
        experiment.run_stage(tmp_path, tmp_path, protocol, "full")
    assert not (tmp_path / "full").exists()


def test_missing_kronos_keeps_rows_and_contract(monkeypatch, tmp_path):
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                          "ticker": ["MRCY", "MRCY"], LABEL_COLUMN: [.1, -.1], "raw": [1., 2.]})
    panel.to_parquet(tmp_path / "raw_panel.parquet")
    (tmp_path / "histories").mkdir()
    pd.DataFrame({"Open": [10., 12.]}, index=pd.DatetimeIndex(panel.date)).to_parquet(tmp_path / "histories/MRCY.parquet")

    def historical(ticker, history, store, contract):
        result = pd.DataFrame(np.nan, index=history.index, columns=KRONOS_COLUMNS)
        result.iloc[0] = [1., 2., 3., 4.]
        return result

    monkeypatch.setattr(experiment, "historical_inputs", historical)
    protocol = {"kronos_contract": {"frozen": True}}
    actual = experiment.materialize(tmp_path, protocol, ["2026-01-02", "2026-01-05"])
    assert actual.index.equals(panel.index)
    assert actual.loc[1, list(KRONOS_COLUMNS)].isna().all()
    assert actual.loc[0, list(KRONOS_COLUMNS)].tolist() == [1., 2., 3., 4.]
    assert actual.attrs[CONTRACT_ATTR] == protocol["kronos_contract"]


def test_matched_folds_share_one_svm_pass_and_final_oof_refit(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = [{"date": date, "ticker": ticker, LABEL_COLUMN: value, "raw": value,
             **dict.fromkeys(PATTERN_EVIDENCE_COLUMNS, 1.), **dict.fromkeys(KRONOS_COLUMNS, 1.)}
            for date in dates for ticker, value in (("A", .1), ("B", -.1), ("H", .05))]
    panel = pd.DataFrame(rows)
    panel.attrs[CONTRACT_ATTR] = {"pinned": True}
    training = panel[panel.ticker.ne("H")]
    protocol = {"tickers": ["A", "B", "H"], "holdout_tickers": ["H"], "through": "2026-11-11",
                "sessions": [str(date.date()) for date in dates], "excluded": ["pruned"],
                "baseline": ["raw"], "candidate": ["raw", *experiment.NEW_COLUMNS],
                "svm_source_features": ["raw", *PATTERN_EVIDENCE_COLUMNS, *KRONOS_COLUMNS],
                "kronos_contract": panel.attrs[CONTRACT_ATTR]}
    experiment.save_json(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(experiment, "N_SPLITS", 2)
    monkeypatch.setattr(experiment, "materialize", lambda *args: panel)
    monkeypatch.setattr(experiment, "ready_contract", lambda *args: (protocol["kronos_contract"], {"ready": True}))
    calls, heads, archives = [], [], {}

    def estimators():
        return {name: SimpleNamespace(feature_names=protocol["svm_source_features"]) for name in STACKED_SVM_COLUMNS}

    def stack(source, **kwargs):
        calls.append(("folds", source.index.tolist(), kwargs))
        assert not source.ticker.eq("H").any()
        for fit_dates, test_dates in ((dates[1:2], dates[2:4]), (dates[1:4], dates[4:6])):
            base = source[source.date.isin(fit_dates)]
            stacked = base.assign(**dict.fromkeys(STACKED_SVM_COLUMNS, .5))
            test = source[source.date.isin(test_dates)].assign(**dict.fromkeys(STACKED_SVM_COLUMNS, .5))
            yield StackedSvmFold(base, stacked, test, estimators())

    def train(frame, specs):
        columns = sorted(specs[0].included_features)
        heads.append((columns, frame.index.tolist()))
        member = SimpleNamespace(feature_names=columns, estimator=SimpleNamespace(
            feature_importance=lambda kind: np.ones(len(columns))))
        return Ensemble([member], [1.])

    def fit(source, **kwargs):
        calls.append(("final", source.index.tolist(), kwargs))
        assert not source.ticker.eq("H").any()
        return estimators()

    monkeypatch.setattr(experiment, "iter_stacked_svm_folds", stack)
    monkeypatch.setattr(experiment, "fit_stacked_svm_estimators", fit)
    monkeypatch.setattr(experiment, "train_ensemble", train)
    monkeypatch.setattr(experiment, "predict", lambda model, frame: frame.raw.to_numpy())
    monkeypatch.setattr(experiment, "predict_ensemble", lambda model, frame: frame.raw.to_numpy())
    monkeypatch.setattr(experiment, "ModelStore", lambda directory: SimpleNamespace(write=lambda name, value: archives.update({name: value})))
    result = experiment.run_stage(tmp_path, tmp_path, protocol, "full")
    assert [call[0] for call in calls] == ["folds", "final"]
    assert calls[1][1] == training.index.tolist()
    assert calls[0][2]["outputs"] == STACKED_SVM_COLUMNS
    assert calls[0][2]["excluded_features"] == {"pruned"}
    assert all(heads[i][1] == heads[i + 1][1] for i in (0, 2, 4))
    final_indices = training[training.date.isin(dates[1:])].index.tolist()
    assert heads[-1][1] == final_indices
    assert len(archives["final_candidate"].stacked_svm_estimators) == 7
    assert archives["final_baseline"].stacked_svm_estimators is None
    assert result["final_head_rows"] == len(final_indices)
    assert result["holdout_tickers"]["candidate"]["days"] == 4
    assert all(value == 0 for value in result["training_tickers"]["paired_candidate_minus_baseline"]["top1"]["ci95"])


def test_wait_stops_when_backfill_dies_or_has_unresolved_errors(monkeypatch, tmp_path):
    protocol = {"tickers": ["A"], "through": "2026-11-11", "kronos_contract": {"pinned": True}}
    monkeypatch.setattr(experiment, "KronosStore", lambda: SimpleNamespace(collection_contract=lambda: protocol["kronos_contract"]))
    process = tmp_path / "process.json"
    experiment.save_json(process, {"pid": 123})
    monkeypatch.setattr(experiment, "ready_contract", lambda *args: (None, {"ready": False}))

    def dead(process):
        raise RuntimeError("Kronos backfill stopped before full readiness")

    monkeypatch.setattr(experiment, "check_backfill_process", dead)
    with pytest.raises(RuntimeError, match="stopped before full readiness"):
        experiment.wait_for_full(tmp_path, protocol, process, 1)
    monkeypatch.setattr(experiment, "ready_contract", lambda *args: (None, {"statuses": {"worker_error": 1}}))
    with pytest.raises(RuntimeError, match="unresolved failures"):
        experiment.wait_for_full(tmp_path, protocol, process, 1)


def test_process_guard_rejects_zombie_reused_pid_and_deadline(monkeypatch):
    started = datetime.now(timezone.utc)
    process = {"started_at": started.isoformat(), "pid": 123,
               "command": ["caffeinate", "-i", "kronos", "--max-seconds", "21600"]}
    stamp = started.astimezone().strftime("%a %b %d %H:%M:%S %Y")
    observed = SimpleNamespace(returncode=0, stdout=f"{stamp} S python kronos.runfiles/_kronos_stage2_bootstrap.py --max-seconds 21600")
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: observed)
    experiment.check_backfill_process(process)
    observed.stdout = f"{stamp} Z caffeinate -i kronos --max-seconds 21600"
    with pytest.raises(RuntimeError, match="stopped before full readiness"):
        experiment.check_backfill_process(process)
    observed.stdout = f"{stamp} S unrelated-worker"
    with pytest.raises(RuntimeError, match="process identity changed"):
        experiment.check_backfill_process(process)
    process["started_at"] = (started - timedelta(hours=7)).isoformat()
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        experiment.check_backfill_process(process)


def test_reports_from_another_frozen_run_cannot_be_overwritten(monkeypatch, tmp_path):
    runtime = tmp_path / ".runtime/research" / experiment.EXPERIMENT
    reports = tmp_path / "research" / experiment.EXPERIMENT
    runtime.mkdir(parents=True)
    reports.mkdir(parents=True)
    experiment.save_json(runtime / "protocol.json", {})
    experiment.save_json(reports / "frozen_protocol.json", {"runtime_protocol_sha256": "another-run"})
    monkeypatch.setattr(experiment, "data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(experiment, "checked_protocol", lambda runtime: {})
    monkeypatch.setattr(experiment, "run_stage", lambda *args: pytest.fail("must not fit"))
    monkeypatch.setattr(sys, "argv", ["rank_input_experiment", "--mode", "pilot"])
    with pytest.raises(ValueError, match="choose fresh --runtime AND --reports"):
        experiment.main()
