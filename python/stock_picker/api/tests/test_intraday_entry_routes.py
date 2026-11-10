from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from stock_picker.api.app import app


def test_entry_train_conflict_and_invalid_list():
    with TestClient(app) as client:
        with patch("stock_picker.api.routes.intraday_entry_job", SimpleNamespace(start=lambda: False)):
            assert client.post("/api/intraday-entry/training").status_code == 409
        assert client.post("/api/intraday-entry/predict", json={"tickers": ["A"]*21}).status_code == 422


def test_entry_model_and_prediction_wiring():
    summary = dict(ready=False, model_name="intraday_entry", version="1.0", evidence_status="experimental",
                   feature_count=0, features=[], dataset={}, notes=[], default_tickers=["A"])
    output = dict(session="2026-11-10", generated_at="2026-11-10T15:00:00Z", elapsed_ms=10,
                  evidence_status="experimental", predictions=[])
    fake = SimpleNamespace(summary=lambda: summary, predict=lambda names: output)
    with TestClient(app) as client, patch("stock_picker.api.routes.get_intraday_entry_service", return_value=fake):
        response = client.get("/api/intraday-entry/model")
        assert response.status_code == 200
        assert response.json()["default_tickers"] == ["A"]
        assert client.post("/api/intraday-entry/predict", json={"tickers": ["A"]}).json()["elapsed_ms"] == 10
