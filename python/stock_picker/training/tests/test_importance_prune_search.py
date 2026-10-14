import numpy as np
import pandas as pd
import pytest

from stock_picker.features.pruning import RESEARCH_CANDIDATE_COLUMNS
from stock_picker.training import importance_prune_search as search
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import TrainedModel


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-09-08", periods=16)
    rows = pd.DataFrame({
        "date": dates,
        "ticker": "TRAIN",
        LABEL_COLUMN: np.linspace(-0.01, 0.01, len(dates)),
        "prior_close": 10.0,
    })
    for name in RESEARCH_CANDIDATE_COLUMNS:
        rows[name] = 0.5
    return rows


class _GainEstimator:
    def __init__(self, names):
        self.names = names

    def feature_importance(self, importance_type):
        assert importance_type == "gain"
        return [0 if name == "prior_close" else 1 for name in self.names]


def test_lowest_gain_breaks_ties_by_name_and_can_prune_old_feature():
    names = ["new_b", "prior_close", "new_a"]
    model = TrainedModel("lightgbm_rank", _GainEstimator(names), names)
    dropped = search.lowest_gain_features(model, 2)
    assert [item["name"] for item in dropped] == ["prior_close", "new_a"]
    assert [item["gain"] for item in dropped] == [0.0, 1.0]


def test_selection_is_refit_inside_each_training_fold(monkeypatch):
    monkeypatch.setattr(search.UniverseStore, "first_seen_by_ticker", lambda self: {"TRAIN": "2026-09-08"})
    monkeypatch.setattr(search.PrunedFeatureStore, "read", lambda self: set())
    trained = []

    def fake_train(frame, *, included_features, **_):
        trained.append((frame["date"].max(), set(included_features)))
        names = sorted(included_features)
        return TrainedModel("lightgbm_rank", _GainEstimator(names), names)

    monkeypatch.setattr(search, "train_lightgbm_rank", fake_train)
    monkeypatch.setattr(search, "predict", lambda _model, frame: frame[LABEL_COLUMN].to_numpy())
    report = search.compare_prunes(_frame(), rounds=2, folds=3)

    assert report["full_feature_count"] == len(RESEARCH_CANDIDATE_COLUMNS) + 1
    assert len(report["folds"]["0"]) == 3
    assert [fold["feature_count"] for fold in report["folds"]["5"]] == [
        len(RESEARCH_CANDIDATE_COLUMNS) - 4,
    ] * 3
    assert all(fold["dropped"][0]["name"] == "prior_close" for fold in report["folds"]["5"])
    assert len(trained) == 9
    assert all(train_max < pd.Timestamp(fold["test_from"]) for (train_max, _), fold in
               zip(trained[::3], report["folds"]["0"]))


def test_research_artifact_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "rank_gain_prune_3_full_2026-09-29.pkl"
    path.touch()
    with pytest.raises(FileExistsError):
        search.train_pruned_research_model(_frame(), count=3, output_dir=tmp_path)
