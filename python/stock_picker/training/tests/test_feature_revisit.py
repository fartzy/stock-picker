import numpy as np
import pandas as pd

from stock_picker.features.pattern_evidence import PATTERN_EVIDENCE_COLUMNS
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.feature_revisit import candidate_variants, choose_variant, daily_results, date_windows


def test_restoration_keeps_baseline_and_only_eligible_features():
    variants = candidate_variants({"kept"}, {"old", "zero"}, {"old": 2, "zero": 0, "bad": 100})
    assert variants["restore_all_eligible"] == {"kept", "old", "zero"}
    assert variants["restore_single1"] == {"kept", "old"}
    assert "restore_single2" not in variants
    assert all("bad" not in columns for columns in variants.values())
    for name in PATTERN_EVIDENCE_COLUMNS:
        assert variants[f"add_{name}"] == {"kept", name}


def test_later_confirmation_is_disjoint_from_all_selection_dates():
    dates = pd.bdate_range("2025-01-01", periods=251)
    screen, confirmation = date_windows(dates)
    assert len(screen) == 3
    assert screen[-1][1] < confirmation[0]
    assert confirmation[-1] == dates[-1]
    assert all(start <= end for start, end in screen)
    assert len(dates[(dates >= confirmation[0]) & (dates <= confirmation[1])]) == 21


def test_selection_requires_consistent_rank_one_and_no_top_five_damage():
    base = [{"top1": 0.01, "top5": 0.01}] * 3
    results = {"baseline": base,
               "one_lucky_day": [{"top1": 1, "top5": 1}, base[0], base[0]],
               "hurts_top5": [{"top1": 0.03, "top5": 0}] * 3,
               "consistent": [{"top1": 0.02, "top5": 0.01}] * 3}
    assert choose_variant(results) == "consistent"
    assert choose_variant({"baseline": base, "hurts_top5": results["hurts_top5"]}) == "baseline"


def test_daily_top_one_uses_predicted_order_with_ticker_ties_and_zero_cost():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"] * 3),
                          "ticker": ["B", "A", "C"], LABEL_COLUMN: [-0.1, 0.2, 0.9]})
    result = daily_results(frame, np.array([1.0, 1.0, 0.0])).iloc[0]
    assert result.top1 == 0.2
    assert result.top5 == np.mean([-0.1, 0.2, 0.9])
