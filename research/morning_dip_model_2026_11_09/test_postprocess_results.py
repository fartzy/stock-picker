"""Small independent arithmetic/selection checks, without model fits."""
import unittest

import postprocess_results as audit

import numpy as np
import pandas as pd


class ResultAuditTests(unittest.TestCase):
    def test_fit_mae_uses_every_stock_day_and_never_rank_scores(self):
        # First two stocks can share a date; MAE must still weight all3 rows
        # equally rather than averaging a two-stock day and a one-stock day.
        actual = np.array([.01, -.02, .03])
        predicted = np.array([0., 0., .06])
        result = audit.fit_error_diagnostic("fit", predicted, actual)
        self.assertAlmostEqual(result["stock_day_mae"], .02)
        self.assertAlmostEqual(result["absolute_error_sum"], .06)
        self.assertEqual(result["scored_stock_days"], 3)
        self.assertIsNone(audit.fit_error_diagnostic("rank", predicted, actual))
        with self.assertRaises(ValueError):
            audit.fit_error_diagnostic("fit", predicted[:-1], actual)

    def test_explicit_input_order_survives_sorted_manifest_family_keys(self):
        protocol = {"families": {"a": ["second"], "z": ["first"]}, "flags": []}
        columns = audit.expected_schema("all", {"added_columns": ["first", "second"]},
                                        ["old"], ["first", "second"], protocol, False)
        self.assertEqual(columns, ["old", "first", "second"])

    def test_scores_control_selection_ties_and_strict_gate(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 6),
                              "ticker": ["F", "E", "D", "C", "B", "A"],
                              audit.TARGET: [.06, .05, .04, .03, .02, .01]})
        daily, selected = audit.score_selections(frame, np.full(6, .005), .005, True)
        self.assertEqual(list(selected["top5"].ticker), list("ABCDE"))
        self.assertEqual(len(selected["gate"]), 0)
        self.assertEqual(daily.gate_mean.iloc[0], 0.)
        self.assertTrue(np.isnan(daily.rank_ic.iloc[0]))

    def test_net_interval_counts_all_cash_days_and_each_strategy_cost(self):
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        left = pd.DataFrame({"gate_mean": [.02, .02, 0, 0, 0], "gate_n": [1, 1, 0, 0, 0]}, index=index)
        right = pd.DataFrame({"gate_mean": [0, 0, 0, 0, .01], "gate_n": [0, 0, 0, 0, 1]}, index=index)
        protocol = {"uncertainty": {"seed": 42, "bootstrap_replications": 20}}
        result = audit.paired_interval(left, right, "gate_mean", protocol, cost_bps=25)
        expected = (2 * (.02 - .0025) - (.01 - .0025)) / 5
        self.assertAlmostEqual(result["mean"], expected)
        np.testing.assert_allclose(result["ci95"], [expected, expected])
        self.assertEqual(result["days"], 5)

    def test_excursions_join_after_selection_and_contributions_reconcile(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
        selected = pd.DataFrame({"date": dates, "ticker": ["A", "B", "A"],
                                 audit.TARGET: [.02, -.01, .03], "score": [3., 2., 4.]})
        outcomes = selected[audit.KEYS].assign(open_to_low_return=[-.04, -.02, -.01],
                                               open_to_high_return=[.03, .02, .04],
                                               low_outcome_valid=True, high_outcome_valid=True,
                                               source_ohlc_consistent=True)
        days = pd.date_range("2024-01-02", periods=3, freq="B")
        risk = audit.excursion_stats(selected, outcomes, days)
        self.assertAlmostEqual(risk["daily_mean_constituent_open_to_low_including_cash"], (-.03 - .01) / 3)
        self.assertEqual(risk["selected_stock_days"], 3)
        self.assertIn("not_portfolio_drawdown", risk)
        concentration = audit.return_concentration(selected, 3)
        summed = sum(row["daily_mean_return_contribution"] for row in concentration["largest_positive_contributors"])
        self.assertAlmostEqual(summed, (.005 + .03) / 3)
        with self.assertRaises(ValueError):
            audit.excursion_stats(selected, outcomes.iloc[:-1], days)


if __name__ == "__main__":
    unittest.main()
