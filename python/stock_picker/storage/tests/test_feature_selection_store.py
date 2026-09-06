from stock_picker.storage.feature_selection_store import FeatureSelectionStore


def test_read_returns_none_before_any_selection(tmp_path):
    store = FeatureSelectionStore(data_dir=tmp_path)

    assert store.read() is None


def test_write_then_read_returns_the_same_set(tmp_path):
    store = FeatureSelectionStore(data_dir=tmp_path)

    store.write({"return_1d", "momentum_spread_5_20d"})

    assert store.read() == {"return_1d", "momentum_spread_5_20d"}


def test_write_none_is_distinct_from_an_empty_set(tmp_path):
    store = FeatureSelectionStore(data_dir=tmp_path)

    store.write(set())
    assert store.read() == set()

    store.write(None)
    assert store.read() is None


def test_clear_resets_to_no_selection(tmp_path):
    store = FeatureSelectionStore(data_dir=tmp_path)
    store.write({"return_1d"})

    store.clear()

    assert store.read() is None


def test_write_persists_across_new_store_instances(tmp_path):
    FeatureSelectionStore(data_dir=tmp_path).write({"return_1d"})

    assert FeatureSelectionStore(data_dir=tmp_path).read() == {"return_1d"}
