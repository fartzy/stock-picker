from stock_picker.features import pruning
from stock_picker.features.dip_context import DIP_COLUMNS


def test_price_pattern_inputs_follow_the_same_explicit_prunes(monkeypatch):
    class FakeStore:
        def read(self):
            return {"previous_quality_prune", "volatility_32d"}

    monkeypatch.setattr(pruning, "PrunedFeatureStore", FakeStore)

    excluded = pruning.pruned_features()

    assert excluded == {"previous_quality_prune", "volatility_32d"}
    assert "market_advance_fraction" not in excluded
    assert "path_cluster_distance" not in excluded
    assert len(pruning.PRICE_PATTERN_COLUMNS) == 19


def test_dip_inputs_have_no_implicit_exclusion_and_follow_explicit_prunes(monkeypatch):
    class FakeStore:
        def read(self):
            return {"dip_deep_unrecovered", "previous_quality_prune"}

    monkeypatch.setattr(pruning, "PrunedFeatureStore", FakeStore)
    assert pruning.pruned_features() & set(DIP_COLUMNS) == {"dip_deep_unrecovered"}
