from stock_picker.storage.jev_outcome_store import JevOutcomeStore
import sqlite3
import pytest


def test_outcomes_append_without_changing_previous_observations(tmp_path):
    store = JevOutcomeStore(data_dir=tmp_path)
    old = {"forecast_id": "one", "observed_at": "2026-11-06T21:15:00+00:00", "issue": "missing"}
    new = {"forecast_id": "one", "observed_at": "2026-11-06T22:15:00+00:00", "exit_price": 10.5}
    store.append(old)
    store.append(new)
    store.append(old)
    assert store.latest() == {"one": new}
    with sqlite3.connect(tmp_path / "outcomes.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError, match="append only"):
            conn.execute("DELETE FROM outcomes")
