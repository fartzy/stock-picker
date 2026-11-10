"""Execution-order checks for the fixed research protocol; no live data needed."""
from datetime import datetime, timedelta
import unittest

from study import ET, evaluate


def stamp(day, hour, minute):
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").replace(tzinfo=ET)


def bar(day, hour, minute, opened=100.0, closed=100.0, low=None):
    return {"t": int(stamp(day, hour, minute).timestamp() * 1000), "o": opened,
            "h": max(opened, closed), "l": min(opened, closed) if low is None else low, "c": closed}


class EntryRules(unittest.TestCase):
    def run_case(self, bars, day="2026-11-09", **kwargs):
        return evaluate(bars, 100.0, stamp(day, 9, 30), stamp(day, 16, 0), **kwargs)

    def test_close_trigger_then_actual_next_open_even_if_it_rebounds(self):
        rows = [bar("2026-11-09", 9, 30, closed=95.0),
                bar("2026-11-09", 9, 31, opened=97.0), bar("2026-11-09", 15, 59, closed=98.0)]
        result = self.run_case(rows)
        self.assertEqual(result["entry_price"], 97.0)
        self.assertAlmostEqual(result["gross_return"], 98 / 97 - 1)

    def test_low_touch_without_close_trigger_does_not_enter(self):
        rows = [bar("2026-11-09", 9, 30, closed=97.0, low=94.0), bar("2026-11-09", 15, 59)]
        self.assertEqual(self.run_case(rows)["status"], "no_entry")

    def test_missing_next_minute_never_retries_later(self):
        rows = [bar("2026-11-09", 9, 30, closed=95.0), bar("2026-11-09", 9, 32, closed=94.0),
                bar("2026-11-09", 9, 33, opened=93.0), bar("2026-11-09", 15, 59)]
        self.assertEqual(self.run_case(rows)["reason"], "missing_immediately_next_entry_minute")

    def test_noon_entry_is_outside_morning_window(self):
        rows = [bar("2026-11-09", 9, 30), bar("2026-11-09", 11, 59, closed=95.0),
                bar("2026-11-09", 12, 0, opened=94.0), bar("2026-11-09", 15, 59)]
        self.assertEqual(self.run_case(rows)["status"], "no_entry")
        self.assertEqual(self.run_case(rows, before_noon=False)["entry_price"], 94.0)

    def test_partial_publication_minute_is_not_used(self):
        rows = [bar("2026-11-09", 9, 30, closed=94.0), bar("2026-11-09", 9, 32, closed=95.0),
                bar("2026-11-09", 9, 33, closed=95.0), bar("2026-11-09", 9, 34, opened=94.0),
                bar("2026-11-09", 15, 59)]
        result = self.run_case(rows, publication=stamp("2026-11-09", 9, 32) + timedelta(seconds=30))
        self.assertIn("09:34:00", result["entry_at"])

    def test_missing_final_exit_minute_is_unknown_not_cash(self):
        rows = [bar("2026-11-09", 9, 30, closed=95.0), bar("2026-11-09", 9, 31),
                bar("2026-11-09", 15, 58, closed=120.0)]
        result = self.run_case(rows)
        self.assertEqual(result["reason"], "missing_expected_final_exit_minute")
        self.assertNotIn("gross_return", result)

    def test_session_boundaries_and_dst(self):
        for day, offset in (("2026-10-30", "-04:00"), ("2026-11-02", "-05:00")):
            rows = [bar(day, 9, 29, closed=90.0), bar(day, 9, 30), bar(day, 9, 40, closed=95.0),
                    bar(day, 9, 41, opened=94.0), bar(day, 15, 59, closed=100.0), bar(day, 16, 0, closed=50.0)]
            result = self.run_case(rows, day)
            self.assertTrue(result["entry_at"].endswith(offset))
            self.assertEqual(result["exit_price"], 100.0)

    def test_nonfinite_bar_cannot_supply_an_entry(self):
        rows = [bar("2026-11-09", 9, 30, closed=95.0), bar("2026-11-09", 9, 31),
                bar("2026-11-09", 15, 59)]
        rows[1]["o"] = float("nan")
        result = self.run_case(rows)
        self.assertEqual(result["reason"], "invalid_or_nonfinite_minute_data")
        self.assertEqual(result["invalid_bars"], 1)

    def test_invalid_earlier_close_cannot_be_skipped_for_later_signal(self):
        rows = [bar("2026-11-09", 9, 30), bar("2026-11-09", 9, 31),
                bar("2026-11-09", 9, 32, closed=95.0), bar("2026-11-09", 9, 33),
                bar("2026-11-09", 15, 59)]
        rows[1]["c"] = float("nan")
        self.assertEqual(self.run_case(rows)["reason"], "invalid_or_nonfinite_minute_data")


if __name__ == "__main__":
    unittest.main()
