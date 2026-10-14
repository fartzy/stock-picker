from stock_picker.features import pruning


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
