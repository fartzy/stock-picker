"""Bad local history must not abort a whole candidate benchmark."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import exchange_calendars as xcals
import pandas as pd

from tools.kronos.benchmark import ROOT, local_requests, main

sys.path.insert(0, str(ROOT / "python"))
from stock_picker.storage.universe_store import UniverseStore


def history():
    calendar = xcals.get_calendar("XNYS")
    dates = calendar.sessions_window(calendar.previous_session("2025-01-21"), -128)
    return pd.DataFrame({"Open": 100., "High": 102., "Low": 98., "Close": 101., "Volume": 1000.}, index=dates)


class BenchmarkTests(unittest.TestCase):
    def test_missing_corrupt_and_incomplete_schema_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            UniverseStore(root / "universe").sync({name: "test" for name in ("BROKEN", "MISSING", "SCHEMA", "VALID")})
            (root / "prices").mkdir()
            (root / "prices" / "BROKEN.parquet").write_bytes(b"corrupt parquet")
            history()[["Open"]].to_parquet(root / "prices" / "SCHEMA.parquet")
            valid_path = root / "prices" / "VALID.parquet"
            history().to_parquet(valid_path)
            requests, rejected, hashes = local_requests("2025-01-21", 20, root)
            self.assertEqual([r["ticker"] for r in requests], ["VALID"])
            self.assertEqual({r["ticker"] for r in rejected}, {"BROKEN", "MISSING", "SCHEMA"})
            self.assertTrue(all(r["reason"] for r in rejected))
            self.assertEqual(hashes["VALID"], hashlib.sha256(valid_path.read_bytes()).hexdigest())
            self.assertEqual(len(requests[0]["bars"]), 128)

    def test_no_eligible_history_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            UniverseStore(root / "universe").sync({"MISSING": "test"})
            arguments = ["benchmark", "--target-session", "2025-01-21", "--data-root", str(root), "--output", str(root / "report.json")]
            with patch.object(sys, "argv", arguments), patch("tools.kronos.benchmark.DailyWorker") as loader:
                with self.assertRaisesRegex(ValueError, "no stocks"):
                    main()
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
