"""Meaningful execution, information-timing and zero-expense research checks."""
from datetime import datetime, timedelta
import unittest

from analyze import (ET, PROTOCOL, anchor_payload, at_clock, checkpoint, cohort_label, context,
                     create_cases, first_event, ms, prepare_session, split_issue, summarize,
                     weekly_mean_interval)


def time(day, clock):
    return datetime.fromisoformat(f"{day}T{clock}:00").replace(tzinfo=ET)


def bar(clock, opened=100., closed=None, day="2026-11-09", low=None):
    closed = opened if closed is None else closed
    return {"t": ms(time(day, clock)), "o": opened, "c": closed,
            "h": max(opened, closed), "l": min(opened, closed) if low is None else low}


def session(rows, day="2026-11-09", close="16:00"):
    return prepare_session(rows, time(day, "09:30"), time(day, close))


def event(rows, anchor=100., anchor_name="session_open", open_price=100., **kwargs):
    prepared = session(rows)
    return first_event(prepared, anchor, anchor_name, time("2026-11-09", "09:35"), open_price, **kwargs)


class TimingRules(unittest.TestCase):
    def test_open_price_precedes_rank_and0934close_can_enter0935(self):
        rows = [bar("09:30"), bar("09:34", closed=95), bar("09:35", opened=97), bar("15:59", closed=101)]
        result = event(rows)
        self.assertEqual(result["entry_at"], "2026-11-09T09:35:00-05:00")
        self.assertAlmostEqual(result["price_return"], 101 / 97 - 1)
        self.assertEqual(result["signal_observed_at"], result["entry_at"])

    def test_first_dip_is_distinct_from_later_entry(self):
        rows = [bar("09:30"), bar("09:31", closed=95), bar("09:34", closed=94),
                bar("09:35", opened=93), bar("15:59", closed=101)]
        result = event(rows)
        self.assertIn("09:32:00", result["first_observed_dip_at"])
        self.assertEqual(result["minutes_since_first_observed_dip"], 3)
        self.assertTrue(result["already_below_threshold_at_availability"])

    def test_opening_gap_is_not_fast_intraday_decline(self):
        rows = [bar("09:30", opened=94), bar("09:34", opened=94), bar("09:35", opened=94), bar("15:59", closed=98)]
        result = event(rows, anchor_name="previous_close", open_price=94)
        self.assertEqual(result["onset_bin"], "opening_gap")
        self.assertTrue(result["opening_gap_already_below_threshold"])
        self.assertIn("09:30:00", result["first_observed_dip_at"])

    def test_low_touch_and_preavailability_recovery_do_not_trigger(self):
        rows = [bar("09:30", closed=95), bar("09:34", closed=97, low=94),
                bar("09:35", opened=97), bar("15:59", closed=99)]
        self.assertEqual(event(rows)["status"], "no_entry")

    def test_missing_first_eligible_entry_does_not_retry(self):
        rows = [bar("09:30"), bar("09:34", closed=95), bar("09:36", closed=94),
                bar("09:37", opened=94), bar("15:59", closed=102)]
        self.assertEqual(event(rows)["reason"], "missing_immediately_next_entry_minute")

    def test_checkpoint_uses_exact_prior_minute_and_future_onset_is_unknown(self):
        rows = [bar("09:30"), bar("09:34"), bar("09:59", opened=101), bar("10:00", opened=102),
                bar("14:00", closed=95), bar("15:59", closed=105)]
        prepared = session(rows)
        result = checkpoint(prepared, 100, "session_open", time("2026-11-09", "09:35"), 100, "10:00")
        self.assertFalse(result["dip"])
        self.assertIsNone(result["first_observed_dip_at"])
        self.assertIsNone(result["minutes_since_first_observed_dip"])
        self.assertAlmostEqual(result["price_return"], 105 / 102 - 1)

    def test_later_episode_is_distinct_from_first_dip(self):
        rows = [bar("09:30"), bar("09:31", closed=95), bar("09:34", closed=99),
                bar("10:29", closed=95), bar("10:30", opened=95), bar("15:59", closed=100)]
        result = event(rows)
        self.assertEqual(result["entry_time_bin"], "10:30-11:00")
        self.assertTrue(result["recovered_above_threshold_since_first_observed_dip"])
        self.assertIn("10:30:00", result["current_observed_dip_episode_at"])
        self.assertEqual(result["minutes_since_current_observed_dip_episode"], 0)

    def test_missing_checkpoint_signal_is_unknown_not_stale_price(self):
        rows = [bar("09:30"), bar("09:58", closed=95), bar("10:00", opened=95), bar("15:59")]
        result = checkpoint(session(rows), 100, "session_open", time("2026-11-09", "09:35"), 100, "10:00")
        self.assertEqual(result["reason"], "missing_exact_checkpoint_signal_minute")

    def test_halfday_and_noon_are_strict_boundaries(self):
        rows = [bar("09:30"), bar("11:59", closed=95), bar("12:00", opened=95), bar("12:59", closed=100)]
        prepared = session(rows, close="13:00")
        availability = time("2026-11-09", "09:35")
        self.assertEqual(first_event(prepared, 100, "session_open", availability, 100, True)["status"], "no_entry")
        self.assertEqual(first_event(prepared, 100, "session_open", availability, 100)["exit_observed_at"], "2026-11-09T13:00:00-05:00")
        self.assertEqual(checkpoint(prepared, 100, "session_open", availability, 100, "13:00")["reason"], "checkpoint_outside_regular_session")

    def test_no_invalid_early_bar_can_be_skipped(self):
        rows = [bar("09:30"), bar("09:34", closed=95), bar("09:35"), bar("15:59")]
        rows[1]["c"] = float("nan")
        self.assertEqual(event(rows)["reason"], "invalid_or_nonfinite_minute_data")

    def test_missing_exit_does_not_become_cash(self):
        result = event([bar("09:30"), bar("09:34", closed=95), bar("09:35", opened=94)])
        self.assertEqual(result["reason"], "missing_expected_final_exit_minute")
        self.assertNotIn("price_return", result)

    def test_dst_and_afterhours_are_excluded(self):
        for day, offset in (("2026-10-30", "-04:00"), ("2026-11-02", "-05:00")):
            rows = [bar("09:29", opened=80, day=day), bar("09:30", day=day), bar("09:34", closed=95, day=day),
                    bar("09:35", opened=94, day=day), bar("15:59", closed=99, day=day), bar("16:00", closed=50, day=day)]
            result = first_event(session(rows, day), 100, "session_open", time(day, "09:35"), 100)
            self.assertEqual(result["exit_price"], 99)
            self.assertTrue(result["entry_at"].endswith(offset))

    def test_split_at_previous_open_already_in_prior_close(self):
        row = {"ticker": "TEST", "previous_session": "2026-11-06", "session": "2026-11-09"}
        metadata = {"complete": True, "relevant_events": [{"ticker": "TEST", "execution_date": "2026-11-06"}]}
        self.assertIsNone(split_issue(metadata, row))
        metadata["relevant_events"][0]["execution_date"] = "2026-11-09"
        self.assertEqual(split_issue(metadata, row), "split_crosses_previous_close_reference")

    def test_open_basis_conflict_cannot_create_a_dip(self):
        row = {"cohort": "live", "session": "2026-11-09", "previous_session": "2026-11-06", "ticker": "TEST", "rank": 1,
               "rank_score": .5, "open_at": time("2026-11-09", "09:30").isoformat(), "close_at": time("2026-11-09", "16:00").isoformat()}
        daily = {("TEST", day): {"adjusted": False, "payload": {"status": "OK", "symbol": "TEST", "from": day, "open": 100., "close": 100.}}
                 for day in ("2026-11-06", "2026-11-09")}
        doc = {("TEST", "2026-11-09"): {"bars": [bar("09:30", opened=90), bar("09:34", opened=90), bar("09:35", opened=90), bar("15:59", opened=95)]}}
        events, checkpoints = create_cases([row], daily, doc, {"complete": True, "relevant_events": []})
        self.assertTrue(all(r["reason"] == "daily_minute_open_price_basis_conflict" for r in events + checkpoints))

    def test_provider_alias_must_be_verified(self):
        key = ("MOG-B", "2026-11-09")
        record = {"adjusted": False, "provider_ticker": "MOG.B", "payload": {"status": "OK", "from": key[1], "symbol": "MOG.B", "open": 100.}}
        self.assertIsNone(anchor_payload({key: record}, *key, "open"))
        record["provider_alias_verified"] = True
        self.assertEqual(anchor_payload({key: record}, *key, "open"), 100.)


class EvidenceCounting(unittest.TestCase):
    def row(self, day, value, ticker="TEST"):
        return {"session": day, "ticker": ticker, "status": "completed", "signal": True,
                "price_return": value, "worst_observed_further_return": min(0, value)}

    def test_zero_expense_keeps_tiny_positive_return_positive(self):
        result = summarize([self.row("2026-11-09", .00001)])
        self.assertEqual(result["mean_return"], .00001)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(PROTOCOL["expense_fraction"], 0)

    def test_repeated_times_are_not_new_stockdays_or_sessions(self):
        rows = [self.row("2026-11-09", .01) for _ in range(9)]
        result = summarize(rows)
        self.assertEqual(result["observations"], 9)
        self.assertEqual(result["unique_stock_days"], 1)
        self.assertEqual(result["completed_sessions"], 1)
        self.assertIsNone(result["weekly_mean_95pct"]["interval"])

    def test_week_bootstrap_keeps_whole_clusters(self):
        # Every week's two observations average zero; week resampling must
        # retain that exact zero, whereas independent row resampling would not.
        rows = []
        for index in range(8):
            day = (datetime(2026, 9, 7) + timedelta(weeks=index)).date().isoformat()
            rows.extend([self.row(day, -.1, "A"), self.row(day, .1, "B")])
        self.assertEqual(weekly_mean_interval(rows)["interval"], [0., 0.])

    def test_unavailable_is_not_a_cash_opportunity(self):
        rows = [self.row("2026-11-09", .02), {"session": "2026-11-06", "ticker": "A", "status": "no_entry", "signal": False},
                {"session": "2026-11-05", "ticker": "B", "status": "unavailable", "signal": True, "reason": "missing_exit"}]
        result = summarize(rows, True)
        self.assertEqual(result["known_opportunity_denominator"], 2)
        self.assertEqual(result["known_opportunity_mean_return"], .01)

    def test_rank_cohorts_stay_separate(self):
        self.assertEqual(cohort_label({"cohort": "live", "rank": 1}), "live_rank1")
        self.assertEqual(cohort_label({"cohort": "live", "rank": 2}), "live_ranks2_5")
        self.assertEqual(cohort_label({"cohort": "research_baseline", "rank": 1}), "research_baseline_rank1")


if __name__ == "__main__":
    unittest.main()
