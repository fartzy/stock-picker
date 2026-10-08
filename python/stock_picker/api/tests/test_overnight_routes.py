"""Overnight endpoints expose model state and preserve scenario metadata."""

from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from stock_picker.api.app import app
from stock_picker.training.overnight_hold import ObservedNextOpen
from stock_picker.ingestion.massive_overnight import ObservedCurrentTrade

ET = ZoneInfo("America/New_York")


class EmptyModelStore:
    def exists(self, name):
        return False


def test_model_metadata_is_visible_without_an_artifact():
    with patch("stock_picker.api.routes.ModelStore", EmptyModelStore):
        response = TestClient(app).get("/api/overnight/model")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert len(body["feature_columns"]) == 15
    assert body["serving_inputs_pinned"] is False


def test_current_price_prefill_requires_fresh_timestamp_and_valid_ticker():
    now = datetime(2026, 10, 6, 15, 55, tzinfo=ET)
    with patch("stock_picker.api.routes.MassiveOvernightClient") as provider, patch("stock_picker.api.routes.datetime") as clock:
        clock.now.return_value = now
        provider.return_value.fetch_current_trade.return_value = ObservedCurrentTrade(5.915, now, now)
        response = TestClient(app).get("/api/overnight/current-price?ticker=bull")
        assert response.status_code == 200
        assert response.json()["ticker"] == "BULL"
        assert response.json()["price"] == 5.915
        assert datetime.fromisoformat(response.json()["observed_at"].replace("Z", "+00:00")) == now
        assert datetime.fromisoformat(response.json()["session_open_at"]).astimezone(ET) == datetime(2026, 10, 6, 9, 30, tzinfo=ET)
        assert datetime.fromisoformat(response.json()["session_close_at"]).astimezone(ET) == datetime(2026, 10, 6, 16, 0, tzinfo=ET)
        provider.return_value.fetch_current_trade.return_value = ObservedCurrentTrade(
            5.915, now - timedelta(minutes=6), now,
        )
        stale = TestClient(app).get("/api/overnight/current-price?ticker=BULL")
    assert stale.status_code == 503
    assert "five minutes" in stale.json()["detail"]
    assert TestClient(app).get("/api/overnight/current-price?ticker=BAD/SYMBOL").status_code == 422


def test_current_price_rejects_quotes_outside_cash_session_and_keeps_manual_entry():
    client = TestClient(app)
    with patch("stock_picker.api.routes.MassiveOvernightClient") as provider, patch("stock_picker.api.routes.datetime") as clock:
        after_close = datetime(2026, 10, 6, 16, 5, tzinfo=ET)
        clock.now.return_value = after_close
        assert client.get("/api/overnight/current-price?ticker=BULL").status_code == 503
        provider.return_value.fetch_current_trade.assert_not_called()

        before_open = datetime(2026, 10, 6, 9, 29, tzinfo=ET)
        clock.now.return_value = before_open
        assert client.get("/api/overnight/current-price?ticker=BULL").status_code == 503
        provider.return_value.fetch_current_trade.assert_not_called()

        clock.now.return_value = datetime(2026, 10, 10, 15, 55, tzinfo=ET)
        assert client.get("/api/overnight/current-price?ticker=BULL").status_code == 503
        provider.return_value.fetch_current_trade.assert_not_called()

        at_open = datetime(2026, 10, 6, 9, 30, tzinfo=ET)
        clock.now.return_value = at_open
        provider.return_value.fetch_current_trade.return_value = ObservedCurrentTrade(
            5.915, at_open - timedelta(seconds=1), at_open,
        )
        outside_trade = client.get("/api/overnight/current-price?ticker=BULL")
        assert outside_trade.status_code == 503
        assert "manually" in outside_trade.json()["detail"]

        before_close = datetime(2026, 10, 6, 15, 59, 59, tzinfo=ET)
        bell = datetime(2026, 10, 6, 16, 0, tzinfo=ET)
        clock.now.side_effect = [before_close, bell]
        provider.return_value.fetch_current_trade.return_value = ObservedCurrentTrade(
            5.915, before_close, before_close,
        )
        crossed_bell = client.get("/api/overnight/current-price?ticker=BULL")
        assert crossed_bell.status_code == 503
        assert "manually" in crossed_bell.json()["detail"]


def test_current_price_obeys_xnys_early_close():
    client = TestClient(app)
    before_close = datetime(2026, 11, 27, 12, 55, tzinfo=ET)
    after_close = datetime(2026, 11, 27, 13, 5, tzinfo=ET)
    with patch("stock_picker.api.routes.MassiveOvernightClient") as provider, patch("stock_picker.api.routes.datetime") as clock:
        clock.now.return_value = before_close
        provider.return_value.fetch_current_trade.return_value = ObservedCurrentTrade(
            42.10, before_close, before_close,
        )
        live = client.get("/api/overnight/current-price?ticker=BULL")
        assert live.status_code == 200
        assert datetime.fromisoformat(live.json()["session_close_at"]).astimezone(ET) == datetime(2026, 11, 27, 13, 0, tzinfo=ET)
        clock.now.return_value = after_close
        closed = client.get("/api/overnight/current-price?ticker=BULL")
        assert closed.status_code == 503
        provider.return_value.fetch_current_trade.assert_called_once()


def test_forecast_rejects_invalid_symbol_and_reports_missing_model():
    client = TestClient(app)
    assert client.post("/api/overnight/forecast", json={
        "ticker": "BAD/SYMBOL", "assumed_close": 10,
    }).status_code == 422
    with patch("stock_picker.api.routes.serve_overnight_forecast", side_effect=FileNotFoundError("no saved overnight model")):
        response = client.post("/api/overnight/forecast", json={
            "ticker": "WERN", "assumed_close": 32.9,
        })
    assert response.status_code == 503
    assert response.json()["detail"] == "no saved overnight model"


def test_forecast_schema_keeps_assumed_price_and_all_feature_snapshots():
    payload = {
        "ticker": "WERN", "session": "2026-10-06", "today_open": 33.0,
        "last_trade": 32.9, "last_trade_at": "2026-10-06T15:00:00-04:00",
        "quote_fetched_at": "2026-10-06T20:00:00+00:00", "step": 0.05,
        "shares": 100, "day_outputs": {"day_fit_predicted_return": 0.01},
        "model_trained_through": "2026-10-02", "model_label_observed_on": "2026-10-05",
        "model_feature_version": "overnight_with_morning_outputs_v2", "evaluated_rows": 623,
        "cases": [{
            "label": label, "assumed_close": assumed, "predicted_gap": 0.001,
            "projected_open": assumed * 1.001, "difference_per_share": assumed * 0.001,
            "after_cost_difference_per_share": None,
            "gross_difference_for_shares": assumed * 0.1,
            "after_cost_difference_for_shares": None,
            "next_session": "2026-10-07", "oof_model_gap_mae": 0.013,
            "oof_zero_gap_mae": 0.0128, "oof_ticker_mean_gap_mae": 0.014,
            "oof_abs_open_error_p90_at_assumed_price": 1.2,
            "features": {"assumed_close": assumed},
        } for label, assumed in (("primary", 32.9), ("lower", 32.85), ("higher", 32.95))],
    }
    with (
        patch("stock_picker.api.routes.serve_overnight_forecast", return_value=object()),
        patch("stock_picker.api.routes.serialize_result", return_value=payload),
    ):
        response = TestClient(app).post("/api/overnight/forecast", json={
            "ticker": "WERN", "assumed_close": 32.9,
        })
    assert response.status_code == 200
    assert [case["features"]["assumed_close"] for case in response.json()["cases"]] == [32.9, 32.85, 32.95]


def test_what_if_actuals_deduplicate_and_keep_future_open_pending():
    pending = ObservedNextOpen("WERN", date(2026, 10, 6), date(2026, 10, 7), "awaiting_next_open")
    with patch("stock_picker.api.routes.observed_next_open", return_value=pending) as lookup:
        response = TestClient(app).post("/api/what-if/overnight-actuals", json={
            "as_of": "2026-10-06", "tickers": ["wern", "WERN"],
        })
    assert response.status_code == 200
    assert response.json()["rows"] == [{
        "ticker": "WERN", "session": "2026-10-06", "next_session": "2026-10-07",
        "status": "awaiting_next_open", "verified_close": None, "next_open": None, "reason": None,
    }]
    lookup.assert_called_once_with("WERN", date(2026, 10, 6))
    assert TestClient(app).post("/api/what-if/overnight-actuals", json={
        "as_of": "2026-10-06", "tickers": ["BAD/SYMBOL"],
    }).status_code == 422
