from stock_picker.training.restored_feature_search import restoration_variants, top_pruned_features
from stock_picker.training.model import TrainedModel


def test_restoration_only_adds_requested_pruned_columns_without_changing_prune_list():
    available = {"kept", "volatility_32d", "trend_old", "volume_old"}
    pruned = {"trend_old", "volume_old", "unavailable"}
    variants = restoration_variants(available, pruned, {"trend": ["trend_old"], "volume": ["volume_old"]})
    assert variants["baseline"] == {"kept", "volatility_32d"}
    assert variants["restore_trend"] == {"kept", "volatility_32d", "trend_old"}
    assert variants["restore_all"] == available
    assert pruned == {"trend_old", "volume_old", "unavailable"}


def test_top_pruned_candidates_are_positive_gain_and_explicitly_pruned():
    class Estimator:
        def feature_importance(self, importance_type):
            assert importance_type == "gain"
            return [100.0, 3.0, 0.0, 5.0]

    model = TrainedModel("lightgbm_rank", Estimator(), ["kept", "pruned_a", "pruned_zero", "pruned_b"])
    assert top_pruned_features(model, {"pruned_a", "pruned_zero", "pruned_b"}, 1) == {"pruned_b"}
    assert top_pruned_features(model, {"pruned_a", "pruned_zero", "pruned_b"}, 5) == {"pruned_a", "pruned_b"}
