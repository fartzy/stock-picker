import pytest

from stock_picker.storage.model_store import ModelStore
from stock_picker.training import rank_model
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.model import TrainedModel


class Predictor:
    def predict(self, frame):
        return [0.0] * len(frame)


def _ensemble(names, model_type="lightgbm_rank"):
    return Ensemble(members=[TrainedModel(model_type, Predictor(), names)], weights=[1.0])


def test_activation_updates_normal_rank_artifact_and_preserves_previous_model(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_model, "pruned_features", lambda: set())
    live = ModelStore(tmp_path / "models")
    live.write(rank_model.RANK_MODEL_NAME, _ensemble(["old_feature"]))
    source = ModelStore(tmp_path / "source")
    source.write("new_rank", _ensemble(["old_feature", "volatility_32d"]))

    result = rank_model.activate_rank_artifact(source.data_dir / "new_rank.pkl", store=live)

    assert live.read(rank_model.RANK_MODEL_NAME).members[0].feature_names == ["old_feature", "volatility_32d"]
    backup = ModelStore(live.data_dir / "backups").read(result["backup"])
    assert backup.members[0].feature_names == ["old_feature"]


@pytest.mark.parametrize("model_type,features,prunes", [
    ("lightgbm", ["new_feature"], set()),
    ("lightgbm_rank", ["blocked"], {"blocked"}),
])
def test_invalid_or_pruned_artifact_does_not_replace_live_rank(tmp_path, monkeypatch, model_type, features, prunes):
    monkeypatch.setattr(rank_model, "pruned_features", lambda: prunes)
    live = ModelStore(tmp_path / "models")
    live.write(rank_model.RANK_MODEL_NAME, _ensemble(["old_feature"]))
    source = ModelStore(tmp_path / "source")
    source.write("candidate", _ensemble(features, model_type))
    with pytest.raises(ValueError):
        rank_model.activate_rank_artifact(source.data_dir / "candidate.pkl", store=live)
    assert live.read(rank_model.RANK_MODEL_NAME).members[0].feature_names == ["old_feature"]
