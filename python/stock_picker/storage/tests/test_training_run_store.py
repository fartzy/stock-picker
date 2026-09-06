from stock_picker.storage.training_run_store import TrainingRunRecord, TrainingRunStore


def _record(run_id, started_at, status="completed"):
    return TrainingRunRecord(
        run_id=run_id,
        status=status,
        started_at=started_at,
        completed_at=started_at,
        duration_seconds=1.0,
        train_tickers=["AAPL"],
        holdout_tickers=["MSFT"],
        date_range=("2026-01-01", "2026-01-31"),
        resolved_features=["return_1d"],
        model_specs=[{"model_type": "lightgbm", "weight": 1.0, "params": None}],
        fold_metrics=[{"mae": 0.01, "directional_accuracy": 0.5, "n_test_rows": 100}],
        holdout_metrics={"mae": 0.01, "directional_accuracy": 0.5, "n_test_rows": 50},
        threshold_sweep=[{"threshold": 0.005, "n_trades": 10, "hit_rate": 0.6}],
    )


def test_append_then_read_all_round_trips_a_record(tmp_path):
    store = TrainingRunStore(data_dir=tmp_path)
    record = _record("run-1", "2026-01-01T09:00:00-05:00")

    store.append(record)

    [loaded] = store.read_all()
    assert loaded == record


def test_read_all_returns_newest_first(tmp_path):
    store = TrainingRunStore(data_dir=tmp_path)
    store.append(_record("run-1", "2026-01-01T09:00:00-05:00"))
    store.append(_record("run-2", "2026-01-02T09:00:00-05:00"))

    runs = store.read_all()

    assert [r.run_id for r in runs] == ["run-2", "run-1"]


def test_read_all_on_an_empty_store_returns_an_empty_list(tmp_path):
    store = TrainingRunStore(data_dir=tmp_path)

    assert store.read_all() == []


def test_a_failed_run_can_omit_provenance_fields(tmp_path):
    store = TrainingRunStore(data_dir=tmp_path)
    record = TrainingRunRecord(
        run_id="run-failed",
        status="failed",
        started_at="2026-01-01T09:00:00-05:00",
        completed_at="2026-01-01T09:00:05-05:00",
        duration_seconds=5.0,
        error="boom",
    )

    store.append(record)

    [loaded] = store.read_all()
    assert loaded.status == "failed"
    assert loaded.error == "boom"
    assert loaded.train_tickers is None
