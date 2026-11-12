"""No model download is needed for request/provenance validation."""

from copy import deepcopy
from datetime import date, timedelta
import unittest

from tools.kronos.contract import prepare_request, read_manifest, validate_ohlcv


def request():
    first = date(2026, 1, 1)
    return {"ticker": "HURN", "target_session": "2026-11-12", "price_basis": "as_stored",
            "bars": [{"session": (first + timedelta(days=i)).isoformat(), "open": 100, "high": 102,
                      "low": 98, "close": 101, "volume": 1000} for i in range(128)]}


class ContractTests(unittest.TestCase):
    def test_stable_input_and_execution_identity(self):
        sample, manifest = request(), read_manifest()
        a = prepare_request(sample, manifest, "cpu-profile")
        self.assertEqual(a, prepare_request(deepcopy(sample), manifest, "cpu-profile"))
        b = prepare_request(sample, manifest, "mps-profile")
        self.assertEqual(a["input_hash"], b["input_hash"])
        self.assertEqual(a["seed"], b["seed"])
        self.assertNotEqual(a["request_key"], b["request_key"])

    def test_value_and_configuration_revisions_invalidate_key(self):
        sample, manifest = request(), read_manifest()
        original = prepare_request(sample, manifest, "runtime")
        sample["bars"][0]["close"] = 100
        changed = prepare_request(sample, manifest, "runtime")
        self.assertNotEqual(original["input_hash"], changed["input_hash"])
        manifest["parameters"]["sample_count"] = 8
        self.assertNotEqual(changed["request_key"], prepare_request(sample, manifest, "runtime")["request_key"])

    def test_target_outcomes_are_never_inputs(self):
        sample, manifest = request(), read_manifest()
        original = prepare_request(sample, manifest, "runtime")
        sample["actual_target_close"] = 1e9
        self.assertEqual(original, prepare_request(sample, manifest, "runtime"))
        sample["bars"][-1]["session"] = sample["target_session"]
        with self.assertRaisesRegex(ValueError, "strictly before"):
            prepare_request(sample, manifest, "runtime")

    def test_pretraining_overlap_is_rejected(self):
        sample = request()
        sample["target_session"] = "2024-06-28"
        with self.assertRaisesRegex(ValueError, "pretraining"):
            prepare_request(sample, read_manifest(), "runtime")

    def test_duplicate_or_short_windows_are_rejected(self):
        sample = request()
        sample["bars"][-1]["session"] = sample["bars"][-2]["session"]
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_request(sample, read_manifest(), "runtime")
        sample["bars"].pop()
        with self.assertRaisesRegex(ValueError, "lookback"):
            prepare_request(sample, read_manifest(), "runtime")

    def test_invalid_bars_are_not_repaired(self):
        bar = request()["bars"][0]
        for name, value in (("close", float("nan")), ("low", -1), ("high", 90), ("volume", float("inf")), ("volume", 10 ** 400), ("volume", 1e300)):
            bad = {**bar, name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_ohlcv(bad, forecast=True)
        self.assertEqual(bar["high"], 102)

    def test_oversized_json_integer_is_a_validation_error(self):
        sample = request()
        sample["bars"][-1]["volume"] = 10 ** 400
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            prepare_request(sample, read_manifest(), "runtime")


if __name__ == "__main__":
    unittest.main()
