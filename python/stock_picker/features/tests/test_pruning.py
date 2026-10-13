from stock_picker.features import pruning


def test_research_candidates_are_withheld_from_automatic_training(monkeypatch):
    class FakeStore:
        def read(self):
            return {"previous_quality_prune"}

    monkeypatch.setattr(pruning, "PrunedFeatureStore", FakeStore)

    excluded = pruning.pruned_features()

    assert "previous_quality_prune" in excluded
    assert "volatility_32d" in excluded
    assert "market_advance_fraction" in excluded
    assert "path_cluster_distance" in excluded
    assert len(pruning.RESEARCH_CANDIDATE_COLUMNS) == 19
