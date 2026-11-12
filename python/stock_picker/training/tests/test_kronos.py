"""Contract, causality, persistence and operational tests without model weights."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import pickle
import sys

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from stock_picker.features.kronos import (
    CONTRACT_ATTR, KRONOS_COLUMNS, forecast_inputs, historical_inputs, live_inputs,
    ready_contract,
)
from stock_picker.ingestion import kronos as collector
from stock_picker.ingestion.kronos_client import KronosWorkerClient
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.kronos_store import KronosStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset, build_training_frame
from stock_picker.training.ensemble import Ensemble, ModelSpec, ensemble_kronos_contract, predict_ensemble
from stock_picker.training.inference import build_inference_row
from stock_picker.training.live_rows import LiveRow, model_frame
from stock_picker.training.model import TrainedModel, predict
from stock_picker.training.model_features import model_inputs
from stock_picker.training.train import run_walk_forward
from tools.kronos.contract import digest, manifest_digest, read_manifest


def description(device="fake"):
    profile = {"device": device, "packages": {"test": "1"}}
    return {"execution_profile": profile, "execution_fingerprint": digest(profile), "manifest": read_manifest()}


def history():
    index = xcals.get_calendar("XNYS").sessions_in_range("2025-01-02", "2026-02-02")
    close = 100 + np.sin(np.arange(len(index)))
    return pd.DataFrame({"Open": 100., "High": 105., "Low": 95., "Close": close, "Volume": 1000.}, index=index)


def response(request, desc, *, reconstructed=False):
    target = request["target_session"]
    opened = xcals.get_calendar("XNYS").session_open(target)
    generated = opened + pd.Timedelta(minutes=1) if reconstructed else opened - pd.Timedelta(hours=2)
    return {**{name: request[name] for name in ("ticker", "target_session", "input_hash", "request_key", "context_end", "price_basis")},
            "status": "ok", "execution_fingerprint": desc["execution_fingerprint"],
            "manifest_hash": manifest_digest(desc["manifest"]), "generated_at": generated.isoformat(),
            "reconstructed": reconstructed, "forecast": {"open": 101., "close": 102., "high": 104., "low": 98., "volume": 1000.},
            "inference_seconds": .01}


def cached(tmp_path, *, target="2025-12-01", reconstructed=False, device="fake"):
    store = KronosStore(tmp_path)
    desc = description(device)
    contract = store.register_profile(desc)
    store.set_collection_contract(contract)
    prices = history()
    request = collector.request_for_history("TEST", target, prices, desc["manifest"], desc["execution_fingerprint"])
    record = store.save(contract, request, response(request, desc, reconstructed=reconstructed), {"read_at": "real read time"})
    return store, contract, prices, request, record


def test_formula_hand_values_and_uniform_unit_rescaling(tmp_path):
    _, _, _, _, record = cached(tmp_path)
    record["last_context_close"] = 100.
    values = forecast_inputs(record, 100., 100.)
    assert list(values.values()) == pytest.approx([.02, .04, -.02, .06])
    # Old-unit cached forecasts, split-adjusted current prior close/open.
    assert forecast_inputs(record, 50., 50.) == pytest.approx(values)
    scaled = deepcopy(record)
    scaled["last_context_close"] *= 7
    for name in ("open", "high", "low", "close"):
        scaled["forecast"][name] *= 7
    assert forecast_inputs(scaled, 50., 50.) == pytest.approx(values)
    assert all(np.isnan(value) for value in forecast_inputs(record, 100., 100., price_basis="incompatible").values())


def test_cache_preserves_first_generation_and_rejects_conflicts(tmp_path):
    store, contract, _, request, first = cached(tmp_path)
    repeat = response(request, description(), reconstructed=True)
    assert store.save(contract, request, repeat, {"read_at": "later"}) == first
    repeat["forecast"]["close"] = 103.
    with pytest.raises(ValueError, match="conflicting same-key"):
        store.save(contract, request, repeat, {})
    assert store.get(contract, "TEST", request["target_session"])["generated_at"] == first["generated_at"]
    wrong = {**contract, "execution_fingerprint": "another"}
    assert store.get(wrong, "TEST", request["target_session"]) is None
    assert store.get(contract, "TEST", "2025-12-02") is None
    with pytest.raises(ValueError, match="already pinned"):
        store.set_collection_contract(wrong)


def test_target_hlcv_future_mutations_and_training_live_alignment(tmp_path):
    target = "2025-12-01"  # Monday following Thanksgiving/Friday early close.
    store, contract, prices, request, _ = cached(tmp_path, target=target)
    baseline = historical_inputs("TEST", prices, store, contract)
    changed = prices.copy()
    changed.loc[target:, ["High", "Low", "Close", "Volume"]] *= 13
    mutated = historical_inputs("TEST", changed, store, contract)
    pd.testing.assert_series_equal(baseline.loc[target], mutated.loc[target])
    assert request["context_end"] == "2025-11-28"
    live = live_inputs("TEST", target, prices.loc[target, "Open"], request["bars"][-1]["close"],
                       prices.loc[prices.index < target], store, contract)
    assert baseline.loc[target].to_dict() == pytest.approx(live)
    training = build_training_frame(prices, baseline)
    pd.testing.assert_series_equal(training.loc[target, list(KRONOS_COLUMNS)], baseline.loc[target], check_names=False)
    stale = pd.Series(dict.fromkeys(KRONOS_COLUMNS, 123.))
    row = build_inference_row(stale, 100., 100., pd.Timestamp("2025-11-28").date(), pd.Timestamp(target).date())
    assert row[list(KRONOS_COLUMNS)].isna().all().all()


def test_reconstruction_is_historical_only_and_bad_context_stays_unknown(tmp_path):
    store, contract, prices, request, _ = cached(tmp_path, reconstructed=True)
    target = request["target_session"]
    assert historical_inputs("TEST", prices, store, contract).loc[target].notna().all()
    assert all(np.isnan(v) for v in live_inputs("TEST", target, 100., request["bars"][-1]["close"], prices, store, contract).values())
    changed = prices.copy()
    changed.loc[request["context_end"], "Close"] += .01
    assert historical_inputs("TEST", changed, store, contract).loc[target].isna().all()
    assert historical_inputs("ABSENT", prices, store, contract).loc[target].isna().all()
    prices["Stock Splits"] = 0.
    prices.loc[target, "Stock Splits"] = 2.
    assert historical_inputs("TEST", prices, store, contract).loc[target].isna().all()


def test_explicit_counterfactual_replay_uses_reconstruction_without_relabeling_it(tmp_path):
    store, contract, prices, request, record = cached(tmp_path, reconstructed=True)
    target = request["target_session"]
    row = LiveRow("TEST", row=pd.DataFrame({"signal": [1.]}), open_price=100.,
                  prev_close=request["bars"][-1]["close"], as_of=target, prior_history=prices)
    head = Ensemble([TrainedModel("lightgbm", ConstantEstimator(), ["kronos_close"], contract)], [1.])
    assert model_frame([row], head, store)["kronos_close"].isna().all()
    row.historical_replay = True
    replay = model_frame([row], head, store)
    expected = historical_inputs("TEST", prices, store, contract).loc[target, "kronos_close"]
    assert replay.loc[0, "kronos_close"] == pytest.approx(expected)
    assert replay.attrs["kronos_usage"] == "counterfactual_replay"
    assert store.get(contract, "TEST", target) == record
    from stock_picker.training.morning import score_from_quotes
    with pytest.raises(ValueError, match="cannot publish"):
        score_from_quotes({}, persist=True, historical_replay=True)


class FakeClient:
    def __init__(self, fail=False):
        self.starts = self.calls = self.closed = 0
        self.fail = fail

    def start(self):
        self.starts += 1
        return description()

    def predict(self, requests):
        self.calls += 1
        if self.fail:
            raise TimeoutError("transport down")
        return [response(request, description()) for request in requests]

    def close(self):
        self.closed += 1


def test_collector_resume_partial_failure_and_missing_tickers(tmp_path):
    prices = PriceStore(tmp_path / "prices")
    prices.write("TEST", history())
    store = KronosStore(tmp_path / "cache")
    targets = ["2025-12-01", "2025-12-02"]
    first = FakeClient()
    report = collector.collect(["TEST", "CSQR"], targets, prices, store, first, max_requests=1)
    assert first.starts == 1 and first.calls == 1 and first.closed == 1
    assert report["stopped_at_limit"] and not report["coverage"]["ready"]
    failing = FakeClient(fail=True)
    report = collector.collect(["TEST", "CSQR"], targets, prices, store, failing)
    assert failing.starts == 2 and report["counts"]["worker_error"] == 1
    second = FakeClient()
    report = collector.collect(["TEST", "CSQR"], targets, prices, store, second)
    assert report["counts"]["cached"] == 1 and report["counts"]["ok"] == 1
    assert report["coverage"]["ready"] and report["counts"]["missing_context"] == 2
    third = FakeClient()
    collector.collect(["TEST"], targets, prices, store, third)
    assert third.calls == 0  # no new inference or original generation-time overwrite
    # A small successful pilot must not unlock a retained-year feature family.
    assert ready_contract(store, ["TEST", "CSQR"], targets[-1])[0] is None


def test_readiness_unlocks_only_after_the_complete_retained_year(tmp_path):
    prices = PriceStore(tmp_path / "prices")
    index = xcals.get_calendar("XNYS").sessions_in_range("2024-06-03", "2026-02-02")
    data = pd.DataFrame({"Open": 100., "High": 105., "Low": 95., "Close": 101., "Volume": 1000.}, index=index)
    prices.write("TEST", data)
    store = KronosStore(tmp_path / "cache")
    targets = collector.retained_sessions("2026-02-02")
    collector.collect(["TEST", "MISSING"], targets[:-1], prices, store, FakeClient())
    assert ready_contract(store, ["TEST", "MISSING"], targets[-1])[0] is None
    collector.collect(["TEST", "MISSING"], targets[-1:], prices, store, FakeClient())
    contract, coverage = ready_contract(store, ["TEST", "MISSING"], targets[-1])
    assert contract == store.collection_contract() and coverage["ready"]
    assert coverage["completed"] == 2 * len(targets)


def test_worker_persists_between_messages_and_timeout_covers_partial_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv("PYTHONPATH", "/nonexistent/bazel/path")
    monkeypatch.setenv("PYTHONHOME", "/nonexistent/bazel/home")
    script = tmp_path / "fake.py"
    script.write_text("import json,sys\nfor line in sys.stdin:\n x=json.loads(line); print(json.dumps({'execution_fingerprint':'test'} if x.get('describe') else {'results':[{'pid':1} for _ in x['requests']]}),flush=True)\n")
    client = KronosWorkerClient(tmp_path, command=[sys.executable, str(script)], timeout=1)
    assert client.start()["execution_fingerprint"] == "test"
    process = client.process
    assert client.predict([{}]) == [{"pid": 1}]
    assert client.predict([{}, {}]) == [{"pid": 1}, {"pid": 1}]
    assert len(client.predict([{"large": "x" * 400000}])) == 1
    assert client.process is process
    client.close()
    script.write_text("import sys,time\nsys.stdout.write('{');sys.stdout.flush();time.sleep(10)\n")
    client = KronosWorkerClient(tmp_path, command=[sys.executable, str(script)], timeout=.05)
    with pytest.raises(TimeoutError):
        client.start()
    assert client.process is None
    # Ready bytes must not extend the deadline indefinitely without a newline.
    script.write_text("import sys\nwhile True:\n sys.stdout.write(' '*1024);sys.stdout.flush()\n")
    client = KronosWorkerClient(tmp_path, command=[sys.executable, str(script)], timeout=.03)
    with pytest.raises(TimeoutError):
        client.start()
    assert client.process is None


def test_known_target_split_is_unknown_in_historical_live_replay_and_overnight(tmp_path, monkeypatch):
    from stock_picker.training.live_rows import prepare_one, with_live_model_context, frame_for_saved_model
    store, contract, prices, request, _ = cached(tmp_path / "cache", reconstructed=True)
    target = request["target_session"]
    prices["Stock Splits"] = 0.
    prices.loc[target, "Stock Splits"] = 2.
    prices.loc[target, "Open"] = 50.
    assert historical_inputs("TEST", prices, store, contract).loc[target].isna().all()
    ps = PriceStore(tmp_path / "prices")
    ps.write("TEST", prices)
    fs = FeatureStore(tmp_path / "features")
    fs.write("TEST", pd.DataFrame({"signal": 1.}, index=prices.index))
    item = prepare_one("TEST", {"TEST": {"open": 50., "prev_close": request["bars"][-1]["close"]}},
                       set(), fs, ps, pd.Timestamp(target).date(), None, None, blocked=set())
    assert item.known_target_split and item.prior_history.index.max() < pd.Timestamp(target)
    head = Ensemble([TrainedModel("lightgbm", ConstantEstimator(), ["kronos_close"], contract)], [1.])
    assert model_frame([item], head, store)[list(KRONOS_COLUMNS)].isna().all().all()
    item.historical_replay = True
    assert model_frame([item], head, store)[list(KRONOS_COLUMNS)].isna().all().all()
    monkeypatch.setattr("stock_picker.training.live_rows.KronosStore", lambda: store)
    assert frame_for_saved_model(with_live_model_context(item), head)[list(KRONOS_COLUMNS)].isna().all().all()


def test_corrected_recent_source_is_automatically_retried(tmp_path):
    prices = PriceStore(tmp_path / "prices")
    data = history()
    data.loc["2025-11-28", "High"] = 1.
    prices.write("TEST", data)
    store = KronosStore(tmp_path / "cache")
    targets = ["2025-12-01"]
    assert collector.collect(["TEST"], targets, prices, store, FakeClient())["counts"]["missing_context"] == 1
    prices.write("TEST", history())
    assert collector.collect(["TEST"], targets, prices, store, FakeClient(), refresh_since=targets[0])["counts"]["ok"] == 1
    prior = store.get(store.collection_contract(), "TEST", targets[0])
    data = history()
    data.loc["2025-11-28", "Close"] += .1
    prices.write("TEST", data)
    assert collector.collect(["TEST"], targets, prices, store, FakeClient(), refresh_since=targets[0])["counts"]["ok"] == 1
    revised = store.get(store.collection_contract(), "TEST", targets[0])
    assert revised["request_key"] != prior["request_key"]
    assert store.get(store.collection_contract(), "TEST", targets[0], prior["request_key"]) == prior


def test_actual_parquet_pool_svm_folds_artifact_and_missing_provenance(tmp_path):
    store, contract, prices, _, _ = cached(tmp_path / "cache")
    features = pd.DataFrame({"signal": np.sin(np.arange(len(prices))),
                             "kronos_close": np.cos(np.arange(len(prices))) / 100}, index=prices.index)
    features.attrs[CONTRACT_ATTR] = contract
    fs = FeatureStore(tmp_path / "features")
    fs.write("TEST", features)
    loaded = fs.read("TEST")
    assert loaded.attrs[CONTRACT_ATTR] == contract
    pooled = build_pooled_dataset({"TEST": prices}, {"TEST": loaded})
    assert pooled.attrs[CONTRACT_ATTR] == contract
    for outputs in ({"svc_direction_margin"}, {"svc_direction_margin", "svr_oof_pred"}):
        specs = [ModelSpec("lightgbm", params={"num_boost_round": 2, "num_threads": 1},
                           included_features={"signal", *outputs})]
        result = run_walk_forward(pooled, specs=specs, n_splits=2, tracking_dir=tmp_path / "mlruns",
                                  stack_direction_margin=True, refit_on_all_data=True,
                                  rank_spec=ModelSpec("lightgbm_rank", params={"num_boost_round": 2, "num_threads": 1},
                                                      included_features={"signal", *outputs}))
        for model in (result[-1].final_model, result[-1].final_rank_model):
            assert "kronos_close" not in model.members[0].feature_names
            assert ensemble_kronos_contract(model) == contract  # saved SVM consumes it
            archive = ModelStore(tmp_path / "models")
            archive.write("test", model)
            restored = archive.read("test")
            assert np.isfinite(predict_ensemble(restored, pooled)).all()
            if len(outputs) == 2:
                from stock_picker.training.overnight_day_scores import align_overnight_morning_inputs
                from stock_picker.training.overnight_model import score_day_model_frame
                aligned = align_overnight_morning_inputs(pooled)
                assert aligned.attrs[CONTRACT_ATTR] == contract
                assert np.isfinite(score_day_model_frame(aligned, result[-1].final_model, result[-1].final_rank_model)).all().all()
            bad = pooled.copy()
            bad.attrs.clear()
            with pytest.raises(ValueError, match="Kronos"):
                predict_ensemble(restored, bad)
            bad.attrs[CONTRACT_ATTR] = {**contract, "execution_fingerprint": "wrong"}
            with pytest.raises(ValueError, match="Kronos"):
                predict_ensemble(restored, bad)
    loaded.attrs.clear()
    with pytest.raises(ValueError, match="Kronos"):
        build_pooled_dataset({"TEST": prices}, {"TEST": loaded})


class ConstantEstimator:
    def predict(self, frame):
        return np.zeros(len(frame))


class KronosEstimator:
    def predict(self, frame):
        return frame["kronos_close"].to_numpy()

    def decision_function(self, frame):
        return self.predict(frame)


def test_overnight_live_fit_rank_svm_and_variants_use_each_saved_contract(tmp_path, monkeypatch):
    from stock_picker.training.live_rows import with_live_model_context
    from stock_picker.training.overnight_model import DAY_OUTPUT_COLUMNS, score_day_model_frame
    from stock_picker.training.overnight_variants import VARIANT_OUTPUT_COLUMNS
    store, contract, prices, request, _ = cached(tmp_path)
    other_description = description("other")
    other = store.register_profile(other_description)
    other_request = collector.request_for_history("TEST", request["target_session"], prices,
                                                  other_description["manifest"], other["execution_fingerprint"])
    other_result = response(other_request, other_description)
    other_result["forecast"]["close"] = 103.
    store.save(other, other_request, other_result, {})
    monkeypatch.setattr("stock_picker.training.live_rows.KronosStore", lambda: store)
    def model(kind, pinned):
        return TrainedModel(kind, KronosEstimator(), ["kronos_close"], pinned)
    fit = Ensemble([model("lightgbm", contract)], [1.], stacked_svm_estimators={
        "svr_oof_pred": model("svr", contract), "svc_direction_margin": model("svc_direction", contract)})
    rank = Ensemble([model("lightgbm_rank", other)], [1.])
    variants = {name: Ensemble([model("lightgbm", other)], [1.]) for name in VARIANT_OUTPUT_COLUMNS}
    target = request["target_session"]
    item = LiveRow("TEST", row=pd.DataFrame({"signal": [1.]}), open_price=100.,
                   prev_close=request["bars"][-1]["close"], as_of=target, prior_history=prices)
    scored = score_day_model_frame(with_live_model_context(item), fit, rank, variants).iloc[0]
    expected_a = forecast_inputs(store.get(contract, "TEST", target), 100., item.prev_close)["kronos_close"]
    expected_b = forecast_inputs(store.get(other, "TEST", target), 100., item.prev_close)["kronos_close"]
    assert scored[DAY_OUTPUT_COLUMNS[0]] == pytest.approx(expected_a)
    assert scored[DAY_OUTPUT_COLUMNS[1]] == pytest.approx(expected_b)
    assert scored[DAY_OUTPUT_COLUMNS[2]] == pytest.approx(expected_a)
    assert scored[DAY_OUTPUT_COLUMNS[3]] == pytest.approx(expected_a)
    assert all(scored[name] == pytest.approx(expected_b) for name in VARIANT_OUTPUT_COLUMNS)


def test_old_pickles_and_per_head_contracts(tmp_path):
    legacy = TrainedModel("lightgbm", ConstantEstimator(), ["signal"])
    del legacy.kronos_contract
    legacy = pickle.loads(pickle.dumps(legacy))
    assert list(predict(legacy, pd.DataFrame({"signal": [1.]}))) == [0.]
    assert ensemble_kronos_contract(Ensemble([legacy], [1.])) is None
    store, contract, prices, request, _ = cached(tmp_path / "a")
    target = request["target_session"]
    row = LiveRow("TEST", row=pd.DataFrame({"signal": [1.], "kronos_close": [999.]}),
                  open_price=100., prev_close=request["bars"][-1]["close"], as_of=target, prior_history=prices)
    head = Ensemble([TrainedModel("lightgbm", ConstantEstimator(), ["kronos_close"], contract)], [1.])
    other = Ensemble([TrainedModel("lightgbm_rank", ConstantEstimator(), ["kronos_close"],
                                  {**contract, "execution_fingerprint": "other"})], [1.])
    assert model_frame([row], head, store)["kronos_close"].notna().all()
    assert model_frame([row], other, store)["kronos_close"].isna().all()
    assert row.row.loc[0, "kronos_close"] == 999.  # shared base is not mutated
    inputs = {item["key"]: item for item in model_inputs(fit=head, rank=other)}
    assert inputs["fit"]["model_outputs"] == ["kronos_close"]
    assert inputs["rank"]["model_outputs"] == ["kronos_close"]


def test_nightly_optional_collection_order_and_failure(monkeypatch, tmp_path):
    from stock_picker.training import nightly, news_ingest
    from stock_picker.ingestion import fundamentals, weather
    events = []
    monkeypatch.setattr(nightly, "refresh_prices", lambda: events.append("prices"))
    monkeypatch.setattr(nightly, "refresh_completed_paper_book_day", lambda _: [])
    monkeypatch.setattr(news_ingest, "refresh_candidate_news", lambda _: None)
    monkeypatch.setattr(news_ingest, "ingest_universe_news", lambda: None)
    monkeypatch.setattr(fundamentals, "refresh_missing_sectors", lambda: 0)
    monkeypatch.setattr(weather, "refresh_nyc_weather", lambda: None)
    monkeypatch.setattr(nightly, "rebuild_features", lambda: events.append("features"))
    monkeypatch.setattr(nightly, "persist_training", lambda: events.append("training"))
    def failed(**kwargs):
        events.append("kronos")
        raise TimeoutError("optional worker down")
    monkeypatch.setattr(collector, "collect_daily", failed)
    assert nightly.run_nightly() == 0
    assert events == ["prices", "kronos", "features", "training"]


def test_clean_clone_daily_collection_never_starts_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "data_root", lambda: tmp_path / "data")
    assert collector.collect_daily()["status"] == "unavailable"


def test_cli_relative_paths_are_anchored_to_invocation_directory(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    runfiles = tmp_path / "runfiles"
    runfiles.mkdir()
    monkeypatch.chdir(runfiles)
    monkeypatch.setattr(collector, "data_root", lambda: root / "data")
    seen = {}
    def fake_collect(tickers, targets, prices, store, client, **kwargs):
        seen.update(cache=store.directory, prices=prices._data_dir, runtime=client.root)
        return {"counts": {"worker_error": 0, "invalid_request": 0}}
    monkeypatch.setattr(collector, "collect", fake_collect)
    monkeypatch.setattr(sys, "argv", ["kronos", "--start", "2026-11-10", "--end", "2026-11-12",
                                     "--tickers", "TEST", "--cache-dir", "cache", "--price-dir", "prices",
                                     "--runtime-root", "runtime", "--report", "report.json"])
    collector.main()
    assert seen == {"cache": root / "cache", "prices": root / "prices", "runtime": root / "runtime"}
    assert (root / "report.json").exists()
    assert not (runfiles / "report.json").exists()


def test_invalid_forecast_is_retained_but_never_repaired(tmp_path):
    store, contract, prices, request, _ = cached(tmp_path)
    next_request = collector.request_for_history("TEST", "2025-12-02", prices, description()["manifest"], contract["execution_fingerprint"])
    bad = response(next_request, description())
    bad.update(status="invalid_forecast", validation_error="incoherent forecast OHLC")
    bad["forecast"]["high"] = 1.
    saved = store.save(contract, next_request, bad, {})
    assert saved["forecast"]["high"] == 1.
    assert all(np.isnan(value) for value in forecast_inputs(saved, 100., 100.).values())
    assert historical_inputs("TEST", prices, store, contract).loc["2025-12-02"].isna().all()
