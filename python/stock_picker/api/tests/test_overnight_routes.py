"""Overnight endpoints expose model state and preserve scenario metadata."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from stock_picker.api.app import app


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
