import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from stock_picker.storage.jev_forecast_store import JevForecastStore


def record(identity="one"):
    return {"id": identity, "ticker": "IONS", "session": "2026-11-09",
            "requested_at": "2026-11-09T16:00:00+00:00", "state": {"entry": 28.12}}


def test_forecast_identity_cannot_be_overwritten_or_deleted(tmp_path):
    store = JevForecastStore(tmp_path)
    saved = record()
    store.append(saved)
    store.append(saved)
    saved["state"]["entry"] = 32.27
    with pytest.raises(ValueError, match="immutable"):
        store.append(saved)
    assert store.read("one")["state"]["entry"] == 28.12
    with store._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM forecasts")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE forecasts SET ticker = 'OTHER'")


def test_concurrent_requests_and_repeated_assessments_are_all_retained(tmp_path):
    store = JevForecastStore(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.append(record(str(i))), range(20)))
    assert len(store.list_records(ticker="IONS", session="2026-11-09")) == 20
    assert store.list_records(ticker="OTHER") == []
    assert store.read("missing") is None


def test_store_rejects_non_json_probabilities_and_missing_identity(tmp_path):
    store = JevForecastStore(tmp_path)
    with pytest.raises(ValueError):
        store.append({**record(), "probability": float("nan")})
    with pytest.raises(ValueError, match="requires id"):
        store.append({**record(), "id": ""})
