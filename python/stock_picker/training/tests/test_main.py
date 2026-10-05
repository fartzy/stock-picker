from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from stock_picker.storage.training_run_store import TrainingRunStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN
from stock_picker.training.ensemble import ModelSpec, predict_ensemble
from stock_picker.training.main import TrainingSummary, _date_range, main, run_training
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


def test_run_training_never_fits_svc_on_holdout_and_scores_raw_holdout(tmp_path, monkeypatch):
    from stock_picker.training import direction_stack
    from stock_picker.training import train as training_module

    dates = pd.bdate_range("2026-01-02", periods=40)
    pooled = pd.DataFrame([
        {
            "ticker": ticker,
            "date": date,
            "signal": float(day % 5 + ticker_index / 10),
            LABEL_COLUMN: 0.01 if (day + ticker_index) % 2 else -0.01,
        }
        for ticker_index, ticker in enumerate(("AAA", "BBB", "HOLD"))
        for day, date in enumerate(dates)
    ])
    written = {}
    write_order = []
    fitted_tickers = []
    rank_calls = []
    actual_fit = direction_stack.train_svc_direction
    actual_walk_forward = training_module.run_walk_forward

    def capture_fit(frame, **kwargs):
        fitted_tickers.append(set(frame["ticker"]))
        return actual_fit(frame, **kwargs)

    def isolated_walk_forward(frame, **kwargs):
        assert kwargs["stack_direction_margin"] is True
        return actual_walk_forward(frame, tracking_dir=tmp_path / "mlflow", **kwargs)

    def record_write(name, model):
        write_order.append(name)
        written[name] = model

    monkeypatch.setattr(direction_stack, "train_svc_direction", capture_fit)
    monkeypatch.setattr(
        "stock_picker.training.main.UniverseStore",
        lambda: SimpleNamespace(active_tickers=lambda: ["AAA", "BBB", "HOLD"]),
    )
    monkeypatch.setattr("stock_picker.training.main.select_holdout_tickers", lambda _: {"HOLD"})
    monkeypatch.setattr(
        "stock_picker.training.main._load_pooled_dataset",
        lambda tickers, *_: pooled[pooled["ticker"].isin(tickers)].copy(),
    )
    monkeypatch.setattr(
        "stock_picker.training.main.pruned_features",
        lambda: set(STACKED_SVM_COLUMNS) - {DIRECTION_MARGIN_COLUMN},
    )
    monkeypatch.setattr("stock_picker.training.main.ModelStore", lambda: SimpleNamespace(write=record_write))
    monkeypatch.setattr("stock_picker.training.main.run_walk_forward", isolated_walk_forward)
    monkeypatch.setattr("stock_picker.training.main.train_logistic_regression", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "stock_picker.training.rank_model.train_and_persist_rank_model",
        lambda **kwargs: rank_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "stock_picker.training.main.sweep_thresholds",
        lambda *_args, **_kwargs: pd.DataFrame([{"threshold": 0.005, "n_trades": 1, "hit_rate": 0.5}]),
    )

    summary = run_training(
        model_specs=[
            ModelSpec("lightgbm", params={"min_data_in_leaf": 2}),
            ModelSpec("lightgbm_rank"),
        ],
        run_id="example",
    )

    assert fitted_tickers and all(names == {"AAA", "BBB"} for names in fitted_tickers)
    assert summary.holdout_tickers == ["HOLD"]
    assert DIRECTION_MARGIN_COLUMN in summary.resolved_features
    archived = written["day_session_return_example"]
    assert archived is written["day_session_return"]
    assert archived.direction_svc is not None
    assert rank_calls == [{"included_features": None}]  # Rank remains a separate raw-feature path.
    assert write_order == [
        "day_session_return_logistic_diagnostic",
        "day_session_return_example",
        "day_session_return",
    ]
    raw_holdout = pooled[pooled["ticker"] == "HOLD"]
    assert len(predict_ensemble(archived, raw_holdout)) == len(raw_holdout)


def _stub_fold_training(monkeypatch, holdout_tickers):
    """Isolate publication order without fitting real models or touching data files."""
    pooled = pd.DataFrame({
        "ticker": ["TRAIN"],
        "date": pd.to_datetime(["2026-01-02"]),
        "signal": [1.0],
        LABEL_COLUMN: [0.01],
    })
    latest = object()
    stored = {"day_session_return": latest}
    write_order = []
    model = SimpleNamespace(members=[SimpleNamespace(feature_names=["signal"])], direction_svc=None)
    result = SimpleNamespace(
        fold=0,
        model=model,
        metrics=EvaluationMetrics(mae=0.01, directional_accuracy=0.5, n_test_rows=1),
        train_rows=1,
    )
    monkeypatch.setattr(
        "stock_picker.training.main.UniverseStore",
        lambda: SimpleNamespace(active_tickers=lambda: ["TRAIN", "HOLD"]),
    )
    monkeypatch.setattr("stock_picker.training.main.select_holdout_tickers", lambda _: holdout_tickers)
    monkeypatch.setattr("stock_picker.training.main.PriceStore", lambda: object())
    monkeypatch.setattr("stock_picker.training.main.FeatureStore", lambda: object())
    monkeypatch.setattr("stock_picker.training.main.pruned_features", lambda: set())
    monkeypatch.setattr("stock_picker.training.main._load_pooled_dataset", lambda *_: pooled)
    monkeypatch.setattr("stock_picker.training.main.run_walk_forward", lambda *_args, **_kwargs: [result])
    monkeypatch.setattr("stock_picker.training.main.train_logistic_regression", lambda *_args, **_kwargs: object())
    def record_write(name, value):
        write_order.append(name)
        stored[name] = value

    monkeypatch.setattr("stock_picker.training.main.ModelStore", lambda: SimpleNamespace(write=record_write))
    return stored, latest, write_order


def test_failed_holdout_never_overwrites_latest_or_creates_archive(monkeypatch):
    stored, latest, write_order = _stub_fold_training(monkeypatch, {"HOLD"})

    def fail_holdout(*_args, **_kwargs):
        raise RuntimeError("holdout failed")

    monkeypatch.setattr(
        "stock_picker.training.main.evaluate_ensemble",
        fail_holdout,
    )

    with pytest.raises(RuntimeError, match="holdout failed"):
        run_training(model_specs=[ModelSpec("lightgbm")], run_id="failed")

    assert stored == {"day_session_return": latest}
    assert write_order == []


def test_no_holdout_still_publishes_archive_before_latest(monkeypatch):
    stored, latest, write_order = _stub_fold_training(monkeypatch, set())

    summary = run_training(model_specs=[ModelSpec("lightgbm")], run_id="noholdout")

    assert summary.holdout_metrics is None
    assert summary.threshold_sweep is None
    assert stored["day_session_return"] is stored["day_session_return_noholdout"]
    assert stored["day_session_return"] is not latest
    assert write_order == [
        "day_session_return_logistic_diagnostic",
        "day_session_return_noholdout",
        "day_session_return",
    ]


def test_failed_rank_companion_does_not_publish_new_fit(monkeypatch):
    stored, latest, write_order = _stub_fold_training(monkeypatch, set())

    def fail_rank(**_kwargs):
        raise RuntimeError("Rank failed")

    monkeypatch.setattr(
        "stock_picker.training.rank_model.train_and_persist_rank_model", fail_rank
    )

    with pytest.raises(RuntimeError, match="Rank failed"):
        run_training(
            model_specs=[ModelSpec("lightgbm"), ModelSpec("lightgbm_rank")],
            run_id="failed_rank",
        )

    assert stored == {"day_session_return": latest}
    assert write_order == []


@pytest.mark.parametrize(
    "included,pruned",
    [
        ({DIRECTION_MARGIN_COLUMN}, set()),
        ({DIRECTION_MARGIN_COLUMN, "pruned_signal"}, {"pruned_signal"}),
    ],
)
def test_derived_only_positive_selection_fails_before_training_or_publication(
    monkeypatch, included, pruned
):
    monkeypatch.setattr(
        "stock_picker.training.main.UniverseStore",
        lambda: SimpleNamespace(active_tickers=lambda: ["TRAIN", "HOLD"]),
    )
    monkeypatch.setattr("stock_picker.training.main.select_holdout_tickers", lambda _: {"HOLD"})
    monkeypatch.setattr("stock_picker.training.main.PriceStore", lambda: object())
    monkeypatch.setattr("stock_picker.training.main.FeatureStore", lambda: object())
    monkeypatch.setattr("stock_picker.training.main.pruned_features", lambda: pruned)
    monkeypatch.setattr(
        "stock_picker.training.main.run_walk_forward",
        lambda *_args, **_kwargs: pytest.fail("should reject before fitting"),
    )
    monkeypatch.setattr(
        "stock_picker.training.main.ModelStore",
        lambda: pytest.fail("should reject before creating the model store"),
    )

    with pytest.raises(ValueError, match="at least one unpruned raw feature"):
        run_training(included_features=included, model_specs=[ModelSpec("lightgbm")])


def test_catalog_only_raw_feature_cannot_bypass_post_load_selection_guard(monkeypatch):
    stored, latest, write_order = _stub_fold_training(monkeypatch, set())
    monkeypatch.setattr(
        "stock_picker.training.main.run_walk_forward",
        lambda *_args, **_kwargs: pytest.fail("should reject before fitting"),
    )

    with pytest.raises(ValueError, match="present in the training dataset"):
        run_training(
            included_features={DIRECTION_MARGIN_COLUMN, "sector_relative_return"},
            model_specs=[ModelSpec("lightgbm")],
        )

    assert stored == {"day_session_return": latest}
    assert write_order == []
