from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime

from fastapi.testclient import TestClient
import pytest

from stock_picker.api.app import app
from stock_picker.paper.jev_outcomes import summarize_jev
from stock_picker.storage.jev_forecast_store import JevForecastStore
from stock_picker.training.jev_shadow import QUESTIONS, shadow_scores


def record():
    return {"id": "one", "schema_version": 1, "question_set_version": "entry_v1", "ticker": "IONS",
            "session": "2026-11-09", "kind": "rank", "list_rank": 1, "score": 0.4,
            "requested_at": "2026-11-09T16:00:00+00:00", "completed_at": "2026-11-09T16:00:01+00:00",
            "evidence_as_of": "2026-11-09T16:00:00+00:00", "target": {"entry_basis": "observed_trade"},
            "state": {}, "questions": {}, "answers": {"net_positive_by_close": 0.6},
            "model": "jev-test", "tokens": 10, "issue": None, "coverage": {"news_status": "degraded"}}


def test_manual_assessment_contract_and_invalid_values():
    client = TestClient(app)
    with patch("stock_picker.api.routes.assess_jev", return_value=record()) as assess:
        result = client.post("/api/jev/assess", json={"ticker": "IONS", "kind": "rank", "entry_price": 28.12,
                                                   "round_trip_cost_per_share": 0.02, "downside_fraction": 0.03})
        assert result.status_code == 200
        assert result.json()["coverage"]["news_status"] == "degraded"
        assert assess.call_args.kwargs["entry_price"] == 28.12
        for bad in ({"entry_price": True}, {"round_trip_cost_per_share": -1}, {"downside_fraction": 1}, {"mode": "order"}):
            assert client.post("/api/jev/assess", json={"ticker": "IONS", "kind": "rank", **bad}).status_code == 422
        assert assess.call_count == 1


def test_unavailable_session_is_explained_without_submitting():
    with patch("stock_picker.api.routes.assess_jev", side_effect=ValueError("Assessments require an open regular XNYS session")):
        response = TestClient(app).post("/api/jev/assess", json={"ticker": "IONS", "kind": "rank"})
    assert response.status_code == 422
    assert "open regular" in response.json()["detail"]


def test_history_and_provider_refresh_do_not_request_new_jev_answers():
    with patch("stock_picker.api.routes.JevForecastStore") as store, patch("stock_picker.api.routes.load_jev_evaluation") as evaluate, patch("stock_picker.api.routes.assess_jev") as assess:
        store.return_value.list_records.return_value = [record()]
        evaluate.return_value = {"forecast_count": 1}
        client = TestClient(app)
        assert client.get("/api/jev/forecasts").json()[0]["id"] == "one"
        assert client.get("/api/jev/evaluation").json()["forecast_count"] == 1
        client.post("/api/jev/evaluation/refresh")
        evaluate.assert_called_with([record()], refresh=True)
        assess.assert_not_called()


@pytest.mark.parametrize("metadata", [
    {"model": {"unexpected": "object"}},
    {"model": "jev-test", "usage": {"input_tokens": "900", "output_tokens": 30}},
    {"model": "jev-test", "usage": {"input_tokens": 900, "output_tokens": -1}},
])
def test_malformed_provider_metadata_cannot_poison_archived_history_or_evaluation(tmp_path, metadata):
    body = {"answers": {name: {"type": "noul", "noul": 0.6} for name in QUESTIONS}, **metadata}
    response = SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)
    provider = SimpleNamespace(post=lambda *args, **kwargs: response)
    pick = SimpleNamespace(ticker="IONS", predicted_return=0.4, open_price=32.27,
                           prev_close=39.38, snapshot_date="2026-11-06", news_check=None)
    store = JevForecastStore(tmp_path)
    scores = shadow_scores([(pick, "rank", 1)], lambda ticker: {}, url="u", api_key="k",
                           session=provider, forecast_store=store)
    forecast_id = scores[("rank", "IONS")]["forecast_id"]
    saved = store.read(forecast_id)
    assert saved["issue"] == "request_failed"
    assert saved["model"] is None and saved["tokens"] is None
    assert all(value is None for value in saved["answers"].values())
    with patch("stock_picker.api.routes.JevForecastStore", return_value=store), patch(
        "stock_picker.api.routes.load_jev_evaluation",
        side_effect=lambda records: summarize_jev(records, {}, now=datetime.fromisoformat(saved["completed_at"])),
    ):
        client = TestClient(app)
        history = client.get("/api/jev/forecasts")
        assert history.status_code == 200
        assert history.json()[0]["id"] == forecast_id
        evaluation = client.get("/api/jev/evaluation")
        assert evaluation.status_code == 200
        assert evaluation.json()["forecast_count"] == 1
        assert evaluation.json()["rows"][0]["status"] == "unavailable"
