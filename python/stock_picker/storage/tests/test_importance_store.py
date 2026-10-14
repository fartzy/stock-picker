import pytest

from stock_picker.storage import importance_store
from stock_picker.storage.importance_store import ImportanceStore


def test_highest_known_survives_lower_values_and_features_missing_from_later_models(tmp_path):
    store = ImportanceStore(tmp_path)
    store.record_many([("old-model", {"pruned_feature": 3.2, "kept": 1.0})])
    store.record_many([("new-model", {"kept": 2.0})])
    store.record_many([("later-model", {"pruned_feature": 0.2})])

    assert store.snapshot()["peaks"] == {
        "pruned_feature": {"value": 3.2, "source": "old-model"},
        "kept": {"value": 2.0, "source": "new-model"},
    }


def test_failed_atomic_publication_preserves_previous_history(tmp_path, monkeypatch):
    store = ImportanceStore(tmp_path)
    store.record_many([("old-model", {"feature": 1.0})])

    def fail_replace(*_args):
        raise OSError("publication failed")

    monkeypatch.setattr(importance_store.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.record_many([("new-model", {"feature": 2.0})])
    assert store.snapshot()["peaks"]["feature"]["value"] == 1.0
