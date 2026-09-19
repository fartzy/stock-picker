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


def test_morning_job_enabled_defaults_true_and_can_be_turned_off(tmp_path):
    store = TrainingConfigStore(data_dir=tmp_path)

    assert store.read().morning_job_enabled is True
    store.write_morning_job_enabled(False)
    assert store.read().morning_job_enabled is False
