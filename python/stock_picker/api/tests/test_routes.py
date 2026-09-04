from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_picker.api.app import app
from stock_picker.features.tests.fixtures import synthetic_history


@pytest.fixture
def client():
    history = synthetic_history(n=140)
    tables = {"AAA": history.assign(x=1.0), "BBB": history.assign(x=2.0)}

    with (
        patch("stock_picker.features.catalog_loader.UniverseStore") as mock_universe_store,
        patch("stock_picker.features.catalog_loader.PriceStore") as mock_price_store,
        patch("stock_picker.features.catalog_loader.FeatureStore") as mock_feature_store,
    ):
        mock_universe_store.return_value.active_tickers.return_value = ["AAA", "BBB"]
        mock_price_store.return_value.read.return_value = history
        mock_feature_store.return_value.read.side_effect = lambda t: tables[t]
        yield TestClient(app)


def test_get_catalog(client):
    response = client.get("/api/catalog")

    assert response.status_code == 200
    body = response.json()
    assert "momentum" in body["catalog"]
    assert "return_1d" in body["descriptions"]


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


def test_get_registry(client):
    response = client.get("/api/registry")

    assert response.status_code == 200
    body = response.json()
    assert body["entities"][0]["name"] == "ticker"
    assert len(body["feature_views"]) == 9
    assert body["feature_services"][0]["name"] == "day_session_return_model"
