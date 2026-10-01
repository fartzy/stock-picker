import stat

import numpy as np
import pandas as pd
import pytest

from stock_picker.storage import model_store
from stock_picker.storage.model_store import ModelStore
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.direction_stack import DIRECTION_MARGIN_COLUMN, score_direction_margin
from stock_picker.training.ensemble import (
    ModelSpec,
    ensemble_feature_names,
    predict_ensemble,
    train_ensemble,
)
from stock_picker.training.model import train_svc_direction


def test_write_then_read_round_trips_a_mixed_ensemble(tmp_path):
    train_frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], LABEL_COLUMN: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    ensemble = train_ensemble(
        train_frame,
        [
            ModelSpec("lightgbm", params={"min_data_in_leaf": 1}),
            ModelSpec("random_forest", params={"n_estimators": 5, "min_samples_leaf": 1}),
        ],
    )

    store = ModelStore(data_dir=tmp_path)
    store.write("test_model", ensemble)
    loaded = store.read("test_model")

    assert np.allclose(predict_ensemble(ensemble, train_frame), predict_ensemble(loaded, train_frame))


def test_exists_reflects_whether_a_model_has_been_written(tmp_path):
    store = ModelStore(data_dir=tmp_path)

    assert store.exists("never_written") is False

    store.write("never_written", {"anything": "picklable"})

    assert store.exists("never_written") is True


def test_replacement_publishes_new_model_without_temporary_file(tmp_path):
    store = ModelStore(data_dir=tmp_path)
    store.write("latest", {"version": 1})
    destination = tmp_path / "latest.pkl"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    destination.chmod(0o640)
    previous_bytes = destination.read_bytes()

    store.write("latest", {"version": 2})

    assert store.read("latest") == {"version": 2}
    assert destination.read_bytes() != previous_bytes
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert set(tmp_path.iterdir()) == {destination}


def test_serialization_failure_preserves_previous_model_and_cleans_temp(tmp_path, monkeypatch):
    store = ModelStore(data_dir=tmp_path)
    store.write("latest", {"version": 1})
    previous_bytes = (tmp_path / "latest.pkl").read_bytes()

    def fail_after_partial_write(_model, file):
        file.write(b"incomplete pickle")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(model_store.pickle, "dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="serialization failed"):
        store.write("latest", {"version": 2})

    assert (tmp_path / "latest.pkl").read_bytes() == previous_bytes
    assert store.read("latest") == {"version": 1}
    assert set(tmp_path.iterdir()) == {tmp_path / "latest.pkl"}


def test_replacement_failure_preserves_previous_model_and_cleans_temp(tmp_path, monkeypatch):
    store = ModelStore(data_dir=tmp_path)
    store.write("latest", {"version": 1})
    previous_bytes = (tmp_path / "latest.pkl").read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(model_store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.write("latest", {"version": 2})

    assert (tmp_path / "latest.pkl").read_bytes() == previous_bytes
    assert store.read("latest") == {"version": 1}
    assert set(tmp_path.iterdir()) == {tmp_path / "latest.pkl"}


def test_direction_svc_round_trips_in_the_same_archive_as_its_lightgbm(tmp_path):
    svc_train = pd.DataFrame({
        "signal": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        LABEL_COLUMN: [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
    })
    raw = pd.DataFrame({
        "signal": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        LABEL_COLUMN: [-0.025, -0.015, -0.005, 0.005, 0.015, 0.025],
    })
    direction_svc = train_svc_direction(svc_train, included_features={"signal"})
    stacked = raw.assign(**{DIRECTION_MARGIN_COLUMN: score_direction_margin(raw, direction_svc)})
    ensemble = train_ensemble(stacked, [ModelSpec(
        "lightgbm", params={"min_data_in_leaf": 1},
        included_features={DIRECTION_MARGIN_COLUMN},
    )])
    ensemble.direction_svc = direction_svc

    store = ModelStore(data_dir=tmp_path)
    store.write("stacked", ensemble)
    loaded = store.read("stacked")

    assert loaded.direction_svc.feature_names == ["signal"]
    assert ensemble_feature_names(loaded) == frozenset({"signal", DIRECTION_MARGIN_COLUMN})
    assert np.allclose(predict_ensemble(loaded, raw), predict_ensemble(ensemble, raw))


def test_legacy_archive_without_direction_svc_field_still_scores(tmp_path):
    raw = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        LABEL_COLUMN: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    ensemble = train_ensemble(raw, [ModelSpec("lightgbm", params={"min_data_in_leaf": 1})])
    del ensemble.direction_svc  # Shape of an archive written before this field existed.
    store = ModelStore(data_dir=tmp_path)
    store.write("legacy", ensemble)

    loaded = store.read("legacy")
    assert "direction_svc" not in vars(loaded)
    assert ensemble_feature_names(loaded) == frozenset({"x"})
    assert np.allclose(predict_ensemble(loaded, raw), predict_ensemble(ensemble, raw))
