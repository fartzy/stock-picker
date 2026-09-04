from stock_picker.storage.feature_exclusion_store import PrunedFeatureStore


def test_read_returns_empty_set_before_any_prune(tmp_path):
    store = PrunedFeatureStore(data_dir=tmp_path)

    assert store.read() == set()


def test_prune_then_read_contains_the_feature(tmp_path):
    store = PrunedFeatureStore(data_dir=tmp_path)

    store.prune("momentum_spread_5_20d")

    assert store.read() == {"momentum_spread_5_20d"}


def test_prune_is_idempotent(tmp_path):
    store = PrunedFeatureStore(data_dir=tmp_path)

    store.prune("momentum_spread_5_20d")
    store.prune("momentum_spread_5_20d")

    assert store.read() == {"momentum_spread_5_20d"}


def test_unprune_removes_the_feature(tmp_path):
    store = PrunedFeatureStore(data_dir=tmp_path)
    store.prune("momentum_spread_5_20d")
    store.prune("return_1d")

    store.unprune("momentum_spread_5_20d")

    assert store.read() == {"return_1d"}


def test_unprune_when_absent_is_a_no_op(tmp_path):
    store = PrunedFeatureStore(data_dir=tmp_path)

    store.unprune("never_pruned")

    assert store.read() == set()
