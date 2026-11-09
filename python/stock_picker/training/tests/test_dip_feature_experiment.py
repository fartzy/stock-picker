"""Small causal/schema/checkpoint tests; no expensive production SVM fits."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.dip_context import DIP_COLUMNS, DIP_FLAG_COLUMNS
from stock_picker.training import dip_feature_experiment as experiment
from stock_picker.training.dataset import LABEL_COLUMN


def panel(days=10):
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    frame = pd.DataFrame({"date": dates.repeat(3), "ticker": ["A", "B", "MRCY"] * days,
                          LABEL_COLUMN: [-.02, 0, .03] * days, "raw": [1., 2., 3.] * days})
    for name in DIP_COLUMNS:
        frame[name] = np.nan
    return frame


def test_temporal_window_uses_exact_prior_dates_and_no_future_labels():
    frame = panel()
    block = {"name": "test", "from": "2024-01-08", "through": "2024-01-12"}
    train, test = experiment.temporal_window(frame, block, 3)
    assert train.date.nunique() == 3
    assert train.date.max() < test.date.min()
    assert train.date.min() == pd.Timestamp("2024-01-03")
    changed = frame.copy()
    changed.loc[changed.date >= block["from"], LABEL_COLUMN] = 1000
    revised, _ = experiment.temporal_window(changed, block, 3)
    pd.testing.assert_frame_equal(train, revised)
    with pytest.raises(ValueError, match="prior sessions"):
        experiment.temporal_window(frame, block, 100)


def test_candidate_missingness_keeps_rows_and_mrcy_but_holdout_fails():
    frame = panel()
    raw = frame[["date", "ticker", LABEL_COLUMN, "raw"]]
    dips = frame[["date", "ticker", *DIP_COLUMNS]]
    aligned = experiment.align_panels(raw, dips, ["raw"], {"HELD"})
    assert len(aligned) == len(raw)
    assert aligned[list(DIP_COLUMNS)].isna().all().all()
    assert aligned.ticker.eq("MRCY").sum() == 10
    with pytest.raises(ValueError, match="holdout"):
        experiment.align_panels(raw, dips, ["raw"], {"A"})
    with pytest.raises(ValueError, match="missing baseline"):
        experiment.align_panels(raw, dips.iloc[1:], ["raw"], set())
    with pytest.raises(ValueError, match="frozen raw"):
        experiment.align_panels(raw.assign(future_close=10), dips, ["raw"], set())


def test_compact_uses_only_positive_training_gain_and_stable_ties():
    names = [name for name in DIP_COLUMNS if name not in DIP_FLAG_COLUMNS]
    gains = {name: 2. for name in names[:10]}
    gains[names[10]], gains[names[11]] = 0., -1.
    compact = experiment.variant_columns(gains)["compact"]
    assert compact[:4] == list(DIP_FLAG_COLUMNS)
    assert compact[4:] == sorted(names[:10])[:8]
    assert len(experiment.variant_columns()["continuous_only"]) == 17


def test_checkpoint_rejects_context_corruption_and_object_arrays(tmp_path):
    manifest = {"schema": 1, "inputs": "abc"}
    run = experiment.Checkpoints(tmp_path / "run", manifest, resume=False)
    context = {"rows": "identity", "fit_end": "2024-01-01"}
    run.save("test", context, np.array([1., 2.]), {"gain": {"raw": 2.}})
    np.testing.assert_array_equal(run.load("test", context)[0], [1., 2.])
    with pytest.raises(ValueError, match="context mismatch"):
        run.load("test", {**context, "rows": "reordered"})
    with pytest.raises(ValueError, match="identical manifest"):
        experiment.Checkpoints(tmp_path / "run", {"inputs": "changed"}, resume=True)
    with pytest.raises(ValueError, match="numeric"):
        run.save("objects", context, np.array(["secret"], dtype=object), {})
    (tmp_path / "run/checkpoints/test.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run.load("test", context)


def test_inner_selection_requires_both_metrics_in_each_block():
    def one(ic, ret, columns=0):
        return {"summary": {"rank_ic": ic, "top20": {"daily_mean": ret}},
                "added_columns": ["x"] * columns}
    results = {block: {name: one(.1, .01) for name in experiment.variant_columns()}
               for block in ["inner_a", "inner_b"]}
    results["inner_a"]["all"] = one(.2, .02, 21)
    results["inner_b"]["all"] = one(.2, .009, 21)
    assert experiment.select_inner(results, list(results))["selected"] == "baseline"
    for block in results:
        results[block]["flags_only"] = one(.11, .011, 4)
    assert experiment.select_inner(results, list(results))["selected"] == "flags_only"
    with pytest.raises(ValueError, match="inner blocks"):
        experiment.select_inner({"outer_a": results["inner_a"]}, ["outer_a"])
    results["inner_b"]["flags_only"]["summary"]["rank_ic"] = None
    missing = experiment.select_inner(results, list(results))
    assert missing["selected"] == "baseline"
    assert missing["diagnostics"]["flags_only"]["paired_ic_top20_by_block"][1][0] is None
    results["inner_a"]["baseline"]["summary"]["rank_ic"] = None
    assert experiment.select_inner(results, list(results))["selected"] == "baseline"


def test_training_projection_is_explicit_and_cache_skips_refit(monkeypatch, tmp_path):
    frame = panel()
    train, test = experiment.temporal_window(frame, {
        "name": "test", "from": "2024-01-08", "through": "2024-01-12"}, 3)
    test["future_close"] = 1000
    calls = []

    def fake_train(frame, **kwargs):
        calls.append(frame)
        assert list(frame.columns) == ["date", "ticker", LABEL_COLUMN, "raw"]
        assert kwargs["included_features"] == {"raw"}
        return SimpleNamespace(feature_names=["raw"], estimator=SimpleNamespace(
            feature_importance=lambda **kwargs: np.array([2.])))

    monkeypatch.setattr(experiment, "train_lightgbm_rank", fake_train)
    monkeypatch.setattr(experiment, "predict", lambda model, frame: frame.raw.to_numpy())
    run = experiment.Checkpoints(tmp_path / "run", {"test": 1}, resume=False)
    protocol = {"rounds": 100, "lightgbm_params": {"num_threads": 4}}
    first = experiment._model_checkpoint(run, "rank_test", train, test, ["raw"], protocol, fit=False)
    second = experiment._model_checkpoint(run, "rank_test", train, test, ["raw"], protocol, fit=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(first[0], second[0])


def test_cash_days_costs_and_paired_week_bootstrap():
    frame = panel(20)
    scores = np.tile([0.0, 0.0, .01], 20)
    scores[:3] = 0
    daily = experiment.daily_metrics(frame, scores)
    summary = experiment.summarize(daily, [0, 10], fit=True)
    assert summary["gate"]["cash_days"] == 1
    assert summary["gate"]["daily_mean"] == pytest.approx(.03 * 19 / 20)
    assert summary["gate"]["cost_sensitivity"]["10"] == pytest.approx(.029 * 19 / 20)
    paired = experiment.paired_week_interval(daily, daily, "top20_mean", 100, 7)
    assert paired["mean"] == 0
    assert paired["ci95"] == [0., 0.]


def test_paired_gate_cost_interval_keeps_cash_days_and_charges_each_strategy():
    # A single ISO week makes every bootstrap replicate analytically identical.
    dates = pd.date_range("2024-01-01", periods=5, freq="B").strftime("%Y-%m-%d")
    baseline = [{"date": when, "gate_mean": 0., "gate_n": 0} for when in dates]
    candidate = [{"date": when, "gate_mean": .02 if i < 2 else 0., "gate_n": 2 if i < 2 else 0}
                 for i, when in enumerate(dates)]
    gross = experiment.paired_week_interval(candidate, baseline, "gate_mean", 100, 7)
    net = experiment.paired_week_interval(candidate, baseline, "gate_mean", 100, 7, cost_bps=25)
    assert gross["mean"] == pytest.approx(.02 * 2 / 5)
    assert net["mean"] == pytest.approx((.02 - .0025) * 2 / 5)
    assert net["ci95"] == pytest.approx([.007, .007])
    assert net["days"] == 5
    # Baseline trades on a different day; its own incurred cost enters the pair.
    baseline[4].update(gate_mean=.01, gate_n=1)
    shifted = experiment.paired_week_interval(candidate, baseline, "gate_mean", 100, 7, cost_bps=25)
    assert shifted["mean"] == pytest.approx((2 * .0175 - .0075) / 5)


def test_full_fit_worker_uses_variant_schema_and_separate_caches(monkeypatch, tmp_path):
    frame = panel()
    raw = frame[["date", "ticker", LABEL_COLUMN, "raw"]]
    dips = frame[["date", "ticker", *DIP_COLUMNS]]
    output = tmp_path / "run"
    manifest = {"test": "full-normal-fit"}
    experiment.Checkpoints(output, manifest, resume=False)
    monkeypatch.setattr(experiment, "_read_baseline", lambda *args: raw.copy())
    monkeypatch.setattr(experiment.pd, "read_parquet", lambda *args, **kwargs: dips.copy())
    columns_seen = []

    def fake_svm(train, *, included_features, outputs):
        columns = [name for name in train if name not in ["date", "ticker", LABEL_COLUMN]]
        assert set(columns) == included_features
        assert train.date.max() < pd.Timestamp("2024-01-08")
        columns_seen.append(columns)
        return {outputs[0]: SimpleNamespace(feature_names=columns)}

    monkeypatch.setattr(experiment, "fit_stacked_svm_estimators", fake_svm)
    monkeypatch.setattr(experiment, "score_stacked_svm", lambda test, estimators, outputs:
                        pd.DataFrame({outputs[0]: np.ones(len(test))}, index=test.index))
    job = {"output": str(output), "manifest": manifest, "block": {
        "name": "oof_test", "from": "2024-01-08", "through": "2024-01-12"},
        "protocol": {"training_sessions": 3}, "baseline": "unused", "dips": "unused",
        "holdout": [], "baseline_raw": ["raw"]}
    base = experiment._svm_block_job({**job, "variant": "baseline", "raw": ["raw"]})
    augmented_columns = ["raw", *DIP_COLUMNS]
    augmented = experiment._svm_block_job({**job, "variant": "all", "raw": augmented_columns})
    assert len(columns_seen) == 14
    assert columns_seen[:7] == [["raw"]] * 7
    assert columns_seen[7:] == [augmented_columns] * 7
    assert base["context"]["test_rows"] == augmented["context"]["test_rows"]
    assert base["context"]["columns"] != augmented["context"]["columns"]
    experiment._svm_block_job({**job, "variant": "all", "raw": augmented_columns})
    assert len(columns_seen) == 14


def test_full_fit_schedules_six_jobs_under_one_worker_cap_and_matches_head_rows(monkeypatch, tmp_path):
    frame = panel()
    protocol = {"training_sessions": 3, "svm_block_workers": 3, "fit_variants": ["baseline", "all"],
                "fit_oof_blocks": [
                    {"name": "oof_a", "from": "2024-01-08", "through": "2024-01-09"},
                    {"name": "oof_b", "from": "2024-01-10", "through": "2024-01-11"}],
                "fit_outer_block": {"name": "outer_a", "from": "2024-01-12", "through": "2024-01-12"}}
    manifest = {"test": "full-fit-pair"}
    run = experiment.Checkpoints(tmp_path / "run", manifest, resume=False)
    seen_jobs = []

    def fake_job(job):
        seen_jobs.append(job)
        train, test = experiment.temporal_window(frame, job["block"], 3)
        context = {"train_rows": experiment.row_digest(train), "test_rows": experiment.row_digest(test),
                   "variant": job["variant"], "columns": job["raw"]}
        for column in experiment.STACKED_SVM_COLUMNS:
            run.save(f"svm_{job['variant']}_{job['block']['name']}_{column}", context,
                     np.full(len(test), len(job["raw"]), dtype=float), {})
        return {"variant": job["variant"], "block": job["block"], "context": context, "rows": len(test)}

    class InlinePool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 3
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def map(self, function, jobs):
            return map(function, jobs)

    def fake_evaluate(run, train, test, raw, protocol, block, *, fit, fit_frames):
        assert fit and set(fit_frames) == {"baseline", "all"}
        base_train, base_test = fit_frames["baseline"]
        new_train, new_test = fit_frames["all"]
        assert experiment.row_digest(base_train) == experiment.row_digest(new_train)
        assert experiment.row_digest(base_test) == experiment.row_digest(new_test)
        assert base_train.svr_oof_pred.eq(1).all()
        assert new_train.svr_oof_pred.eq(22).all()
        assert base_test.svr_oof_pred.eq(1).all()
        assert new_test.svr_oof_pred.eq(22).all()
        assert base_train.date.max() < base_test.date.min()
        return {"checked": True}

    monkeypatch.setattr(experiment, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(experiment, "_svm_block_job", fake_job)
    monkeypatch.setattr(experiment, "evaluate_block", fake_evaluate)
    assert experiment.run_fit(run, frame, ["raw"], protocol, tmp_path / "raw", tmp_path / "dip", set(), manifest) == {"checked": True}
    assert len(seen_jobs) == 6
