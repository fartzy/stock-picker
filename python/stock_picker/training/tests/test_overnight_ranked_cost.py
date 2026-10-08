"""Held-out Rank-pick exit comparisons use verified close/next-open rows."""

from datetime import date

import pandas as pd
import pytest

from stock_picker.ingestion.massive_overnight import VerifiedOvernightBars
from stock_picker.training.overnight_dataset import LABEL_COLUMN, PriceContract
from stock_picker.training.overnight_experiment import (
    ExecutionAssumptions, evaluate_ranked_hold_policy, fetch_verified_experiment_rows,
    labelable_ranked_cohort,
)


def inputs():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    cohort = pd.DataFrame({"date": dates, "ticker": ["AAA", "BBB"]})
    verified = pd.DataFrame({
        "date": [*dates, dates[0]], "ticker": ["AAA", "BBB", "CCC"],
        "assumed_close": [10.0, 20.0, 40.0],
        LABEL_COLUMN: [0.03, -0.05, 0.5],
    })
    predicted = pd.DataFrame({
        "variant": ["core_2", "core_2"], "fold": [1, 1],
        "ticker": ["AAA", "BBB"], "date": dates,
        "train_through": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "assumed_close": [10.0, 20.0], "actual_gap": [0.03, -0.05],
        "actual_next_open": [10.3, 19.0], "predicted_next_open": [10.2, 19.8],
    })
    return predicted, verified, cohort


def test_fetch_keeps_first_requested_label_with_six_prior_closes():
    sessions = pd.to_datetime([
        "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29",
        "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06",
    ])
    history = pd.DataFrame({"Open": 10.0, "Close": 10.0}, index=sessions)
    history.loc["2026-10-05", "Open"] = 10.4
    history.loc["2026-10-06", "Open"] = 10.2
    provenance = pd.DataFrame({
        "source": "massive", "basis": "raw", "action_source": "massive_actions",
        "corporate_action": "verified_none",
    }, index=sessions)

    class FakeClient:
        def __init__(self):
            self.calls = []

        def fetch(self, ticker, start, end):
            self.calls.append((ticker, start, end))
            return VerifiedOvernightBars(history, provenance, (), ())

    client = FakeClient()
    labels, _ = fetch_verified_experiment_rows(
        ["AAA"], date(2026, 10, 2), date(2026, 10, 6), client,
        PriceContract("massive", "raw", "massive_actions"),
    )
    assert client.calls == [("AAA", date(2026, 9, 24), date(2026, 10, 6))]
    assert labels["date"].tolist() == [pd.Timestamp("2026-10-02"), pd.Timestamp("2026-10-05")]
    assert labels[LABEL_COLUMN].tolist() == pytest.approx([0.04, 0.02])


def test_weekend_end_excludes_last_unlabelable_rank_selection_from_coverage():
    cohort = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-01", "2026-10-02"]),
        "ticker": ["AAA", "AAA"],
    })
    labelable = labelable_ranked_cohort(cohort, date(2026, 10, 1), date(2026, 10, 3))
    assert labelable["date"].tolist() == [pd.Timestamp("2026-10-01")]


def test_costed_policy_compares_to_both_fixed_exits_with_equal_capital():
    predicted, verified, cohort = inputs()
    assumptions = ExecutionAssumptions(today_fee_per_share=0.01, next_open_fee_per_share=0.01)
    rows, summary = evaluate_ranked_hold_policy(predicted, verified, cohort, assumptions)

    assert rows["hold_decision"].tolist() == [True, False]
    assert rows["predicted_hold_edge_per_share"].tolist() == pytest.approx([0.1897, -0.2197])
    assert rows["realized_hold_edge_per_share"].tolist() == pytest.approx([0.28955, -1.0185])
    assert rows["policy_edge_per_share"].tolist() == pytest.approx([0.28955, 0.0])
    assert summary.loc[0, "policy_edge_per_share"] == pytest.approx(0.28955 / 2)
    assert summary.loc[0, "always_hold_edge_per_share"] == pytest.approx((0.28955 - 1.0185) / 2)
    assert summary.loc[0, "always_sell_edge_per_share"] == 0
    assert summary.loc[0, "policy_equal_capital_return"] == pytest.approx((0.28955 / 10) / 2)
    assert summary.loc[0, "always_hold_equal_capital_return"] == pytest.approx((0.28955 / 10 - 1.0185 / 20) / 2)
    assert summary.loc[0, "always_sell_equal_capital_return"] == 0


def test_opening_slippage_can_change_the_decision():
    predicted, verified, cohort = inputs()
    base, _ = evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())
    costly, _ = evaluate_ranked_hold_policy(
        predicted, verified, cohort,
        ExecutionAssumptions(next_open_slippage_bps=300),
    )
    assert base["hold_decision"].tolist() == [True, False]
    assert costly["hold_decision"].tolist() == [False, False]
    assert costly["realized_hold_edge_per_share"].iloc[0] < base["realized_hold_edge_per_share"].iloc[0]


def test_all_fold_summary_weights_each_ticker_session_equally():
    predicted, verified, cohort = inputs()
    predicted.loc[1, "fold"] = 2
    third = predicted.iloc[[0]].copy()
    third["date"] = pd.Timestamp("2026-01-07")
    third["fold"] = 2
    predicted = pd.concat([predicted, third], ignore_index=True)
    third_label = verified.iloc[[0]].copy()
    third_label["date"] = pd.Timestamp("2026-01-07")
    verified = pd.concat([verified, third_label], ignore_index=True)
    cohort = pd.concat([cohort, third_label[["date", "ticker"]]], ignore_index=True)

    rows, summary = evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())
    overall = summary.loc[summary["fold"].eq("all")].iloc[0]
    assert overall["rows"] == 3
    assert overall["policy_edge_per_share"] == pytest.approx(rows["policy_edge_per_share"].mean())
    assert overall["policy_equal_capital_return"] == pytest.approx(rows["policy_equal_capital_return"].mean())


def test_non_rank_or_unverified_prediction_cannot_enter_cost_report():
    predicted, verified, cohort = inputs()
    predicted.loc[0, "ticker"] = "CCC"  # Has a verified label, but was not selected by Rank.
    with pytest.raises(ValueError, match="non-Rank or unverified"):
        evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())
    predicted.loc[0, "ticker"] = "AAA"
    verified = verified.loc[verified["ticker"] != "AAA"]
    with pytest.raises(ValueError, match="non-Rank or unverified"):
        evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())


def test_a_ticker_date_cannot_be_counted_in_two_held_out_folds():
    predicted, verified, cohort = inputs()
    duplicate = predicted.iloc[[0]].copy()
    duplicate["fold"] = 2
    predicted = pd.concat([predicted, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="unique held-out"):
        evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())


def test_prices_and_fit_cutoff_must_match_verified_held_out_rows():
    predicted, verified, cohort = inputs()
    predicted.loc[0, "actual_next_open"] = 10.4
    with pytest.raises(ValueError, match="do not match verified"):
        evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())
    predicted.loc[0, "actual_next_open"] = 10.3
    verified.loc[0, LABEL_COLUMN] = 0.04
    with pytest.raises(ValueError, match="do not match verified"):
        evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())
    verified.loc[0, LABEL_COLUMN] = 0.03
    predicted.loc[0, "train_through"] = predicted.loc[0, "date"]
    with pytest.raises(ValueError, match="strictly held-out"):
        evaluate_ranked_hold_policy(predicted, verified, cohort, ExecutionAssumptions())


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_execution_costs_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ExecutionAssumptions(next_open_fee_per_share=value)
