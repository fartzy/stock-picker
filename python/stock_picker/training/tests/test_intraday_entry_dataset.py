from datetime import datetime
import json
import math

import pandas as pd
import pytest

from stock_picker.features.intraday_entry import FEATURE_COLUMNS, MINUTE_MS, MORNING_FEATURE_COLUMNS
from stock_picker.training.dataset import build_training_frame
from stock_picker.training import intraday_entry_dataset as dataset


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dataset._digest(path)


def _fixture(tmp_path, monkeypatch, *, legacy=False, split=False, missing=(), training_end="2026-11-09"):
    monkeypatch.setattr(dataset, "data_root", lambda: tmp_path / "data")
    study = tmp_path / "study"
    start = datetime.fromisoformat("2026-11-10T14:30:00+00:00")
    opened = round(start.timestamp() * 1000)
    bars = [dict(t=opened + i * MINUTE_MS, o=100 + i / 100, h=101 + i / 100,
                 l=99 + i / 100, c=100.2 + i / 100, v=1000, vw=100 + i / 100)
            for i in range(390) if i not in missing]
    for bar in bars:
        if bar["t"] == opened + 5 * MINUTE_MS:
            bar.update(o=96., h=97., l=95., c=96.1)
        if bar["t"] == opened + 389 * MINUTE_MS:
            bar.update(h=106., c=105.)
    path = tmp_path / "minute.json"
    sha = _write(path, {"ticker": "ABC", "adjusted": False, "bars": bars})
    _write(study / "minute_manifest.json", [dict(ticker="ABC", session="2026-11-10", path=str(path), sha256=sha, complete=True)])
    records = []
    for day in ("2026-11-09", "2026-11-10"):
        payload = dict(status="OK", symbol="ABC", **{"from": day}, open=100., close=101.)
        path = tmp_path / f"daily_{day}.json"
        record = dict(ticker="ABC", date=day, adjusted=False, payload=payload,
                      source_path=str(path), source_sha256=_write(path, payload))
        if not legacy:
            record.update(session=day, complete=True)
        records.append(record)
    _write(study / "daily_prices.json", records)
    row = dict(cohort="research_baseline", ticker="ABC", session="2026-11-10",
               previous_session="2026-11-09", open_at=start.isoformat(),
               close_at="2026-11-10T21:00:00+00:00", training_end=training_end,
               rank=1, rank_score=.75, rank_score_z=1.5,
               morning_features={name: .02 for name in MORNING_FEATURE_COLUMNS})
    # Same stock-day in another experiment must never double the training sample.
    _write(study / "cohort.json", [row, {**row, "cohort": "research_all21"}])
    _write(study / "split_metadata.json", dict(complete=True, coverage_start="2026-11-09",
           coverage_end="2026-11-10", relevant_events=[dict(ticker="ABC", execution_date="2026-11-10")] if split else []))
    _seal(study)
    return study


def _seal(study):
    hashes = {name: dataset._digest(study / name) for name in
              ("cohort.json", "daily_prices.json", "minute_manifest.json", "split_metadata.json")}
    _write(study / "audit_checks.json", {"status": "passed", "artifact_sha256": hashes})


def _build(study):
    return dataset.build_intraday_entry_dataset(study, checkpoints_et=("09:35", "10:00"), expand_cached_top20=False)


@pytest.mark.parametrize("legacy", [False, True])
def test_raw_entry_to_close_label_zero_expense_and_no_duplicate_cohort(tmp_path, monkeypatch, legacy):
    study = _fixture(tmp_path, monkeypatch, legacy=legacy)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = _build(study)
    assert len(result.frame) == 2
    assert result.metadata["candidate_stock_days"] == 1
    assert result.metadata["frozen_audit_verified"]
    assert result.metadata["expense_fraction"] == 0
    first = result.frame.iloc[0]
    assert first["entry_price"] == 96
    assert first["target_return"] == pytest.approx(105 / 96 - 1)
    assert first["return_from_open"] == pytest.approx(-.04)
    assert first["morning_training_end"] < first["session"]
    assert set(FEATURE_COLUMNS).issubset(result.frame)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_split_removes_previous_close_basis_without_discarding_intraday_observation(tmp_path, monkeypatch):
    result = _build(_fixture(tmp_path, monkeypatch, split=True))
    assert len(result.frame) == 2
    assert result.frame.previous_close_known.eq(0).all()
    assert result.frame.return_from_previous_close.isna().all()
    assert result.frame.opening_gap.isna().all()
    assert result.frame.return_from_open.notna().all()


@pytest.mark.parametrize(("missing", "reason", "count"), [
    ((0,), "missing_session_first_minute", 2),
    ((389,), "missing_final_exit_minute", 2),
    ((5,), "missing_exact_entry_minute", 1),
    ((4,), "missing_last_completed_minute", 1),
])
def test_missing_observations_are_excluded_and_counts_reconcile(tmp_path, monkeypatch, missing, reason, count):
    result = _build(_fixture(tmp_path, monkeypatch, missing=missing))
    assert result.metadata["excluded_snapshots"][reason] == count
    assert len(result.frame) + sum(result.metadata["excluded_snapshots"].values()) == 2


def test_same_day_morning_training_and_frozen_source_mutation_are_rejected(tmp_path, monkeypatch):
    study = _fixture(tmp_path, monkeypatch, training_end="2026-11-10")
    with pytest.raises(ValueError, match="trained before"):
        _build(study)
    (study / "cohort.json").write_text("[]")
    with pytest.raises(ValueError, match="frozen cohort.json hash mismatch"):
        _build(study)


def test_prior_morning_context_is_shifted_before_shared_intraday_features(tmp_path, monkeypatch):
    study = _fixture(tmp_path, monkeypatch)
    index = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"])
    history = pd.DataFrame({"Open": [100.] * 4, "Close": [101., 99., 98., 97.]}, index=index)
    features = pd.DataFrame({name: [1., 2., 3., 4.] for name in MORNING_FEATURE_COLUMNS}, index=index)
    prior = build_training_frame(history, features).loc["2026-11-10"]
    assert all(prior[name] == 2 for name in MORNING_FEATURE_COLUMNS)
    changed = features.copy()
    changed.loc["2026-11-10":, :] = 1e9
    after = build_training_frame(history, changed).loc["2026-11-10"]
    rows = dataset._read(study / "cohort.json")
    rows[0]["morning_features"] = {name: float(after[name]) for name in MORNING_FEATURE_COLUMNS}
    _write(study / "cohort.json", rows)
    _seal(study)
    result = _build(study)
    for source, target in MORNING_FEATURE_COLUMNS.items():
        assert result.frame[target].eq(prior[source]).all()
    assert not any("target" in column or "label" in column for column in FEATURE_COLUMNS)
