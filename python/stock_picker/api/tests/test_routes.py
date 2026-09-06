from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_picker.api.app import app
from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.storage.training_run_store import TrainingRunStore

# Prune/unprune mutate a real store instance rather than returning canned
# values, so a stateful fake (shared across both import sites routes.py
# uses) is simpler and more accurate than mock return_value plumbing.
_pruned_state: dict[str, dict] = {}


class _FakePrunedFeatureStore:
    def read(self):
        return set(_pruned_state)

    def read_all(self):
        return list(_pruned_state.values())

    def prune(self, feature, reason="manually pruned"):
        _pruned_state.setdefault(
            feature, {"feature": feature, "reason": reason, "pruned_at": "2026-01-01T00:00:00-05:00"}
        )

    def unprune(self, feature):
        _pruned_state.pop(feature, None)


# Same reasoning as _pruned_state above: mutated by the routes under test, so
# a stateful fake beats mock return_value plumbing.
_training_config_state: dict[str, object] = {"included_features": None, "model_choices": None}


class _FakeTrainingConfig:
    def __init__(self, included_features, model_choices):
        self.included_features = included_features
        self.model_choices = model_choices


class _FakeTrainingConfigStore:
    def read(self):
        return _FakeTrainingConfig(
            _training_config_state["included_features"], _training_config_state["model_choices"]
        )

    def write_included_features(self, included_features):
        _training_config_state["included_features"] = (
            sorted(included_features) if included_features is not None else None
        )

    def write_model_choices(self, model_choices):
        _training_config_state["model_choices"] = model_choices


class _FakeTrainingJob:
    """Stands in for training.job's module-level start()/status() -- never
    touches real walk-forward training (no background thread at all; `start`
    just records what it was called with). `complete()` lets a test move the
    fake straight to a finished state to check the status endpoint's shape."""

    def __init__(self):
        self.status_value = {
            "status": "idle",
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        self.last_included_features = "not called"
        self.last_model_specs = "not called"

    def start(self, included_features=None, model_specs=None):
        if self.status_value["status"] == "running":
            return False
        self.last_included_features = included_features
        self.last_model_specs = model_specs
        self.status_value = {**self.status_value, "status": "running", "started_at": "2026-01-01T00:00:00-05:00"}
        return True

    def complete(self, result):
        self.status_value = {**self.status_value, "status": "completed", "result": result}

    def status(self):
        return self.status_value


_fake_training_job: _FakeTrainingJob | None = None
# A real TrainingRunStore against a per-test tmp_path -- unlike the fakes
# above, this store has no external dependencies worth faking (just local
# file I/O), so the fixture points the real class at a scratch directory
# rather than adding another stateful fake class.
_training_run_store: TrainingRunStore | None = None


@pytest.fixture
def client(tmp_path):
    history = synthetic_history(n=140)
    tables = {"AAA": history.assign(x=1.0), "BBB": history.assign(x=2.0)}

    # Stateful (not a fixed return_value) so test_create_trade can verify the
    # posted trade round-trips through a subsequent GET, same reasoning as
    # the pruned-feature fake above.
    trades_state = [
        {"ticker": "AAA", "side": "buy", "shares": 10, "price": 2.0, "executed_at": "2026-01-01T09:30:00-05:00"},
        {"ticker": "BBB", "side": "buy", "shares": 5, "price": 4.0, "executed_at": "2026-01-02T09:30:00-05:00"},
    ]

    def _append_trade(trade):
        trades_state.append(
            {
                "ticker": trade.ticker,
                "side": trade.side,
                "shares": trade.shares,
                "price": trade.price,
                "executed_at": trade.executed_at,
            }
        )

    def _read_or_missing(ticker):
        if ticker not in tables:
            raise FileNotFoundError(ticker)
        return history

    def _feature_read_or_missing(ticker):
        if ticker not in tables:
            raise FileNotFoundError(ticker)
        return tables[ticker]

    _pruned_state.clear()
    _training_config_state["included_features"] = None
    _training_config_state["model_choices"] = None
    global _fake_training_job
    _fake_training_job = _FakeTrainingJob()
    global _training_run_store
    _training_run_store = TrainingRunStore(data_dir=tmp_path / "training_runs")

    with (
        patch("stock_picker.api.routes.TrainingRunStore", lambda: _training_run_store),
        patch("stock_picker.features.catalog_loader.UniverseStore") as mock_universe_store,
        patch("stock_picker.features.catalog_loader.PriceStore") as mock_price_store,
        patch("stock_picker.features.catalog_loader.FeatureStore") as mock_feature_store,
        patch("stock_picker.api.routes.FeatureStore", mock_feature_store),
        patch("stock_picker.features.price_history.PriceStore") as mock_price_history_store,
        patch("stock_picker.features.price_history.download_price_history") as mock_download_history,
        patch("stock_picker.features.trades.TradeStore") as mock_trade_store,
        patch("stock_picker.api.routes.TradeStore", mock_trade_store),
        patch("stock_picker.features.pruning.PrunedFeatureStore", _FakePrunedFeatureStore),
        patch("stock_picker.api.routes.PrunedFeatureStore", _FakePrunedFeatureStore),
        patch("stock_picker.features.selection.TrainingConfigStore", _FakeTrainingConfigStore),
        patch("stock_picker.api.routes.TrainingConfigStore", _FakeTrainingConfigStore),
        patch("stock_picker.training.ensemble.TrainingConfigStore", _FakeTrainingConfigStore),
        patch("stock_picker.api.routes.training_job", _fake_training_job),
        patch("stock_picker.features.quotes.fetch_quotes") as mock_fetch_quotes,
    ):
        mock_universe_store.return_value.active_tickers.return_value = ["AAA", "BBB"]
        mock_price_store.return_value.read.return_value = history
        mock_feature_store.return_value.read.side_effect = _feature_read_or_missing
        mock_price_history_store.return_value.read.side_effect = _read_or_missing
        mock_download_history.side_effect = lambda tickers, **kw: (
            {tickers[0]: history} if tickers[0] in tables else {}
        )
        mock_trade_store.return_value.read.side_effect = lambda: pd.DataFrame(trades_state)
        mock_trade_store.return_value.append.side_effect = _append_trade
        mock_fetch_quotes.return_value = {"AAA": {"open": 100.0, "last": 105.0}}
        yield TestClient(app)


def test_get_catalog(client):
    response = client.get("/api/catalog")

    assert response.status_code == 200
    body = response.json()
    assert "momentum" in body["catalog"]
    assert "return_1d" in body["descriptions"]
    assert "return_1d" in body["formulas"]


def test_get_coverage(client):
    response = client.get("/api/coverage")

    assert response.status_code == 200
    assert isinstance(response.json()["coverage"], dict)


def test_get_correlation(client):
    response = client.get("/api/correlation")

    assert response.status_code == 200
    body = response.json()
    assert "columns" in body
    assert "matrix" in body
    assert "top_pairs" in body


def test_get_trades(client):
    response = client.get("/api/trades")

    assert response.status_code == 200
    trades = response.json()["trades"]
    assert len(trades) == 2
    assert trades[0]["ticker"] == "BBB"  # newest first
    assert trades[0]["notional"] == 20.0
    assert trades[1]["ticker"] == "AAA"


def test_get_pruned_features_starts_empty(client):
    response = client.get("/api/pruned-features")

    assert response.status_code == 200
    body = response.json()
    assert body["pruned_features"] == []
    assert body["archive"] == []


def test_prune_then_unprune_feature(client):
    prune_response = client.post("/api/features/return_1d/prune")
    assert prune_response.status_code == 200
    prune_body = prune_response.json()
    assert prune_body["pruned_features"] == ["return_1d"]
    assert prune_body["archive"][0]["feature"] == "return_1d"
    assert prune_body["archive"][0]["reason"] == "manually pruned"

    get_response = client.get("/api/pruned-features")
    assert get_response.json()["pruned_features"] == ["return_1d"]

    unprune_response = client.delete("/api/features/return_1d/prune")
    assert unprune_response.status_code == 200
    assert unprune_response.json()["pruned_features"] == []
    assert unprune_response.json()["archive"] == []


def test_prune_feature_with_a_given_reason(client):
    response = client.post(
        "/api/features/return_2d/prune", json={"reason": "high correlation to return_3d (r=0.996)"}
    )

    assert response.status_code == 200
    [entry] = response.json()["archive"]
    assert entry["reason"] == "high correlation to return_3d (r=0.996)"


def test_get_feature_importance_returns_empty_dict_without_a_trained_model(client):
    response = client.get("/api/feature-importance")

    assert response.status_code == 200
    assert response.json() == {"importance": {}, "by_model_type": {}}


def test_get_model_info_returns_empty_list_without_a_trained_model(client):
    response = client.get("/api/model-info")

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_get_model_types_describes_every_known_model_type(client):
    response = client.get("/api/model-types")

    assert response.status_code == 200
    model_types = {m["model_type"] for m in response.json()["model_types"]}
    assert model_types == {"lightgbm", "random_forest", "logistic_regression"}
    for info in response.json()["model_types"]:
        assert info["package"]
        assert info["source_file"].endswith("model.py")


def test_get_training_runs_starts_empty(client):
    response = client.get("/api/training/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_get_training_runs_returns_an_appended_record_newest_first(client):
    from stock_picker.storage.training_run_store import TrainingRunRecord

    _training_run_store.append(
        TrainingRunRecord(
            run_id="run-1",
            status="completed",
            started_at="2026-01-01T09:00:00-05:00",
            completed_at="2026-01-01T09:05:00-05:00",
            duration_seconds=300.0,
            train_tickers=["AAPL"],
            holdout_tickers=["MSFT"],
            date_range=("2026-01-01", "2026-01-31"),
            resolved_features=["return_1d"],
            model_specs=[{"model_type": "lightgbm", "weight": 1.0, "params": None}],
            fold_metrics=[{"mae": 0.01, "directional_accuracy": 0.5, "n_test_rows": 100}],
            holdout_metrics={"mae": 0.01, "directional_accuracy": 0.5, "n_test_rows": 50},
            threshold_sweep=[{"threshold": 0.005, "n_trades": 10, "hit_rate": 0.6}],
        )
    )

    response = client.get("/api/training/runs")

    assert response.status_code == 200
    [run] = response.json()["runs"]
    assert run["run_id"] == "run-1"
    assert run["train_tickers"] == ["AAPL"]
    assert run["date_range"] == ["2026-01-01", "2026-01-31"]


def test_get_feature_selection_starts_as_no_selection(client):
    response = client.get("/api/feature-selection")

    assert response.status_code == 200
    assert response.json() == {"included_features": None}


def test_set_then_get_feature_selection(client):
    post_response = client.post(
        "/api/feature-selection", json={"included_features": ["return_1d", "return_2d"]}
    )
    assert post_response.status_code == 200
    assert post_response.json() == {"included_features": ["return_1d", "return_2d"]}

    get_response = client.get("/api/feature-selection")
    assert get_response.json() == {"included_features": ["return_1d", "return_2d"]}


def test_clear_feature_selection_resets_to_no_selection(client):
    client.post("/api/feature-selection", json={"included_features": ["return_1d"]})

    response = client.delete("/api/feature-selection")

    assert response.status_code == 200
    assert response.json() == {"included_features": None}
    assert client.get("/api/feature-selection").json() == {"included_features": None}


def test_get_model_selection_starts_as_no_selection(client):
    response = client.get("/api/model-selection")

    assert response.status_code == 200
    assert response.json() == {
        "model_choices": None,
        "available_model_types": ["lightgbm", "random_forest"],
    }


def test_set_then_get_model_selection(client):
    post_response = client.post(
        "/api/model-selection", json={"model_choices": [{"model_type": "lightgbm", "weight": 1.0}]}
    )
    assert post_response.status_code == 200
    assert post_response.json()["model_choices"] == [{"model_type": "lightgbm", "weight": 1.0}]

    get_response = client.get("/api/model-selection")
    assert get_response.json()["model_choices"] == [{"model_type": "lightgbm", "weight": 1.0}]


def test_clear_model_selection_resets_to_no_selection(client):
    client.post("/api/model-selection", json={"model_choices": [{"model_type": "lightgbm"}]})

    response = client.delete("/api/model-selection")

    assert response.status_code == 200
    assert response.json()["model_choices"] is None
    assert client.get("/api/model-selection").json()["model_choices"] is None


def test_start_training_run_reports_running_and_forwards_the_selection(client):
    client.post("/api/feature-selection", json={"included_features": ["return_1d"]})
    client.post("/api/model-selection", json={"model_choices": [{"model_type": "lightgbm", "weight": 2.0}]})

    response = client.post("/api/training/run")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert _fake_training_job.last_included_features == {"return_1d"}
    assert [(s.model_type, s.weight) for s in _fake_training_job.last_model_specs] == [("lightgbm", 2.0)]


def test_start_training_run_conflicts_while_already_running(client):
    client.post("/api/training/run")

    response = client.post("/api/training/run")

    assert response.status_code == 409


def test_get_training_status_reflects_a_completed_run(client):
    _fake_training_job.complete(
        {
            "fold_metrics": [{"mae": 0.01, "directional_accuracy": 0.5, "n_test_rows": 100}],
            "holdout_metrics": {"mae": 0.01, "directional_accuracy": 0.55, "n_test_rows": 200},
            "threshold_sweep": None,
            "train_tickers": ["AAPL"],
            "holdout_tickers": ["MSFT"],
            "date_range": ["2026-01-01", "2026-01-31"],
            "resolved_features": ["return_1d"],
            "model_specs": [{"model_type": "lightgbm", "weight": 1.0, "params": None}],
        }
    )

    response = client.get("/api/training/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"]["holdout_metrics"]["mae"] == 0.01


def test_create_trade(client):
    response = client.post("/api/trades", json={"ticker": "CCC", "side": "buy", "shares": 3, "price": 10.0})

    assert response.status_code == 200
    trades = response.json()["trades"]
    assert len(trades) == 3
    assert trades[0]["ticker"] == "CCC"  # newest first
    assert trades[0]["notional"] == 30.0


def test_get_quotes(client):
    response = client.get("/api/quotes", params={"tickers": "AAA"})

    assert response.status_code == 200
    quotes = response.json()["quotes"]
    assert quotes == [
        {
            "ticker": "AAA",
            "open": 100.0,
            "last": 105.0,
            "diff": 5.0,
            "diff_pct": 0.05,
            "prev_close": None,
            "gap": None,
            "gap_pct": None,
        }
    ]


def test_get_positions(client):
    response = client.get("/api/positions")

    assert response.status_code == 200
    positions = {p["ticker"]: p for p in response.json()["positions"]}
    assert positions["AAA"]["closed"] is False
    assert positions["AAA"]["day_open"] == 100.0
    assert positions["AAA"]["pnl"] == round((105.0 - 2.0) * 10, 2)
    # BBB has no matching quote in the fixture -- graceful nulls, not a crash.
    assert positions["BBB"]["current_price"] is None
    assert positions["BBB"]["pnl"] is None


def test_get_price_history_daily(client):
    response = client.get("/api/prices/AAA")

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAA"
    assert body["interval"] == "daily"
    assert len(body["prices"]) == 140
    assert set(body["prices"][0]) == {"date", "open", "high", "low", "close", "volume"}


def test_get_price_history_hourly(client):
    response = client.get("/api/prices/AAA?interval=hourly")

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "hourly"
    assert len(body["prices"]) == 140


def test_get_price_history_404_for_untracked_ticker(client):
    response = client.get("/api/prices/NOT_A_TICKER")

    assert response.status_code == 404


def test_get_feature_values(client):
    response = client.get("/api/features/AAA")

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAA"
    assert "x" in body["columns"]
    assert len(body["rows"]) == 140
    assert set(body["rows"][0]) == {"date", *body["columns"]}


def test_get_feature_values_404_for_untracked_ticker(client):
    response = client.get("/api/features/NOT_A_TICKER")

    assert response.status_code == 404


def test_get_registry(client):
    response = client.get("/api/registry")

    assert response.status_code == 200
    body = response.json()
    assert body["entities"][0]["name"] == "ticker"
    assert len(body["feature_views"]) == 11
    assert body["feature_services"][0]["name"] == "day_session_return_model"
