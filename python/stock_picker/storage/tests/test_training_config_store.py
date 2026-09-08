from stock_picker.storage.training_config_store import ModelChoice, TrainingConfigStore


def test_read_returns_empty_config_before_any_write(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)

    config = store.read()

    assert config.included_features is None
    assert config.model_choices is None
    assert config.selected_run_id is None


def test_write_selected_run_id_then_read_returns_it(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)

    store.write_selected_run_id("abc123")

    assert store.read().selected_run_id == "abc123"


def test_write_selected_run_id_none_resets_to_latest(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)
    store.write_selected_run_id("abc123")

    store.write_selected_run_id(None)

    assert store.read().selected_run_id is None


def test_writing_selected_run_id_does_not_disturb_other_fields(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)
    store.write_included_features({"return_1d"})
    store.write_model_choices([ModelChoice("lightgbm")])

    store.write_selected_run_id("abc123")

    config = store.read()
    assert config.included_features == ["return_1d"]
    assert config.model_choices == [ModelChoice("lightgbm")]
    assert config.selected_run_id == "abc123"


def test_write_included_features_then_read_returns_the_same_set(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)

    store.write_included_features({"return_1d", "momentum_spread_5_20d"})

    assert store.read().included_features == ["momentum_spread_5_20d", "return_1d"]


def test_write_included_features_none_is_distinct_from_an_empty_set(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)

    store.write_included_features(set())
    assert store.read().included_features == []

    store.write_included_features(None)
    assert store.read().included_features is None


def test_write_model_choices_then_read_returns_the_same_choices(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)

    store.write_model_choices([ModelChoice("lightgbm"), ModelChoice("random_forest", weight=0.5)])

    choices = store.read().model_choices
    assert choices == [ModelChoice("lightgbm", weight=1.0), ModelChoice("random_forest", weight=0.5)]


def test_writing_one_field_does_not_disturb_the_other(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)
    store.write_included_features({"return_1d"})

    store.write_model_choices([ModelChoice("lightgbm")])

    config = store.read()
    assert config.included_features == ["return_1d"]
    assert config.model_choices == [ModelChoice("lightgbm")]


def test_write_persists_across_new_store_instances(tmp_path):
    TrainingConfigStore(data_dir=tmp_path).write_included_features({"return_1d"})

    assert TrainingConfigStore(data_dir=tmp_path).read().included_features == ["return_1d"]
