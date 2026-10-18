"""Model inputs come from saved artifacts, so a feature can belong to several models."""

from types import SimpleNamespace

from stock_picker.features.news import NEWS_FEATURE_COLUMNS
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training import model_features as mf


def ensemble(*members):
    return SimpleNamespace(members=[SimpleNamespace(feature_names=list(names)) for names in members])


def by_key(entries):
    return {entry["key"]: entry for entry in entries}


def test_fit_and_rank_read_saved_member_names_and_overlap():
    svm = STACKED_SVM_COLUMNS[0]
    models = by_key(mf.model_inputs(
        fit=ensemble(["rsi_14", "return_5d", svm], ["rsi_14"]),
        rank=ensemble(["rsi_14", "return_5d"]),
    ))
    assert models["fit"]["features"] == sorted(["rsi_14", "return_5d", svm])
    assert models["fit"]["model_outputs"] == [svm]
    assert models["rank"]["features"] == ["return_5d", "rsi_14"]
    assert set(models["fit"]["features"]) & set(models["rank"]["features"]) == {"rsi_14", "return_5d"}


def test_missing_models_are_marked_not_saved():
    models = by_key(mf.model_inputs())
    assert models["fit"]["saved"] is False and models["fit"]["features"] == []
    assert models["overnight"]["saved"] is False
    assert models["overnight"]["features"]  # falls back to the defined input list


def test_overnight_uses_pinned_columns_and_marks_model_outputs():
    overnight = SimpleNamespace(feature_columns=("gap_5d_mean", "day_fit_predicted_return"))
    entry = by_key(mf.model_inputs(overnight=overnight))["overnight"]
    assert entry["features"] == ["gap_5d_mean", "day_fit_predicted_return"]
    assert entry["model_outputs"] == ["day_fit_predicted_return"]


def test_news_lists_article_inputs_and_news_features():
    entry = by_key(mf.model_inputs())["news"]
    assert entry["features"] == list(NEWS_FEATURE_COLUMNS)
    assert entry["articles"]["fields"] == ["headline", "summary"]
    assert entry["articles"]["max_articles"] == 20
