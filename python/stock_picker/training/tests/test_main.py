from unittest.mock import patch

import pandas as pd

from stock_picker.storage.training_run_store import TrainingRunStore
from stock_picker.training.main import TrainingSummary, _date_range, main
from stock_picker.training.model import EvaluationMetrics


def _fake_summary():
    return TrainingSummary(
        fold_metrics=[EvaluationMetrics(mae=0.01, directional_accuracy=0.5, n_test_rows=100)],
        holdout_metrics=EvaluationMetrics(mae=0.02, directional_accuracy=0.55, n_test_rows=200),
        threshold_sweep=[{"threshold": 0.005, "n_trades": 10, "hit_rate": 0.6}],
        train_tickers=["AAPL", "MSFT"],
        holdout_tickers=["GOOG"],
        date_range=("2026-01-01", "2026-01-31"),
        resolved_features=["return_1d"],
        model_specs=[{"model_type": "lightgbm", "weight": 1.0, "params": None}],
    )


def test_main_appends_a_training_run_record_on_success(tmp_path):
    # The CLI entrypoint (bazel run //python/stock_picker/training:main)
    # previously trained and persisted a model without ever recording a run
    # -- silently invisible in Run History even though it really retrained.
    run_store = TrainingRunStore(data_dir=tmp_path)
    with (
        patch("stock_picker.training.main.TrainingRunStore", return_value=run_store),
        patch("stock_picker.training.main.run_training", return_value=_fake_summary()),
        patch("stock_picker.training.main.selected_features", return_value=None),
        patch("stock_picker.training.main.selected_model_specs", return_value=None),
    ):
        main()

    [record] = run_store.read_all()
    assert record.status == "completed"
    assert record.train_tickers == ["AAPL", "MSFT"]
    assert record.holdout_metrics["directional_accuracy"] == 0.55


def test_main_appends_a_failed_training_run_record_and_reraises(tmp_path):
    run_store = TrainingRunStore(data_dir=tmp_path)

    def failing_train(included_features, model_specs, run_id=None):
        raise RuntimeError("boom")

    with (
        patch("stock_picker.training.main.TrainingRunStore", return_value=run_store),
        patch("stock_picker.training.main.run_training", side_effect=failing_train),
        patch("stock_picker.training.main.selected_features", return_value=None),
        patch("stock_picker.training.main.selected_model_specs", return_value=None),
    ):
        try:
            main()
            raised = False
        except RuntimeError:
            raised = True

    assert raised
    [record] = run_store.read_all()
    assert record.status == "failed"
    assert record.error == "boom"


def test_date_range_returns_min_and_max_date_as_strings():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-15", "2026-01-01", "2026-01-31"])})

    assert _date_range(frame) == ("2026-01-01", "2026-01-31")


def test_date_range_handles_a_single_row():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-15"])})

    assert _date_range(frame) == ("2026-01-15", "2026-01-15")
