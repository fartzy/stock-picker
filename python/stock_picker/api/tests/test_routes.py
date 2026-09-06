from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_picker.api.app import app
from stock_picker.features.tests.fixtures import synthetic_history

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


@pytest.fixture
def client():
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

    _pruned_state.clear()

    with (
        patch("stock_picker.features.catalog_loader.UniverseStore") as mock_universe_store,
        patch("stock_picker.features.catalog_loader.PriceStore") as mock_price_store,
        patch("stock_picker.features.catalog_loader.FeatureStore") as mock_feature_store,
        patch("stock_picker.features.price_history.PriceStore") as mock_price_history_store,
        patch("stock_picker.features.price_history.download_price_history") as mock_download_history,
        patch("stock_picker.features.trades.TradeStore") as mock_trade_store,
        patch("stock_picker.api.routes.TradeStore", mock_trade_store),
        patch("stock_picker.features.pruning.PrunedFeatureStore", _FakePrunedFeatureStore),
        patch("stock_picker.api.routes.PrunedFeatureStore", _FakePrunedFeatureStore),
        patch("stock_picker.features.quotes.fetch_quotes") as mock_fetch_quotes,
    ):
        mock_universe_store.return_value.active_tickers.return_value = ["AAA", "BBB"]
        mock_price_store.return_value.read.return_value = history
        mock_feature_store.return_value.read.side_effect = lambda t: tables[t]
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
    assert response.json() == {"importance": {}}


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


def test_get_registry(client):
    response = client.get("/api/registry")

    assert response.status_code == 200
    body = response.json()
    assert body["entities"][0]["name"] == "ticker"
    assert len(body["feature_views"]) == 9
    assert body["feature_services"][0]["name"] == "day_session_return_model"
