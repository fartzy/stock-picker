"""Calendar/protocol tests use the isolated dependencies but no downloaded model."""

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from tools.kronos.contract import read_manifest
from tools.kronos.worker import DailyWorker


class Predictor:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame([dict(open=100, high=103, low=97, close=101, volume=1000, amount=100000)])


def worker():
    instance = object.__new__(DailyWorker)
    instance.manifest = read_manifest()
    instance.execution_fingerprint = "test-profile"
    instance.pd, instance.np = pd, np
    instance.torch = SimpleNamespace(manual_seed=lambda _: None)
    instance.device = "cpu"
    instance.versions = {}
    instance.calendar = xcals.get_calendar("XNYS")
    instance.predictor = Predictor()
    return instance


def request(instance, target="2025-01-21"):
    previous = instance.calendar.previous_session(target)
    dates = instance.calendar.sessions_window(previous, -instance.manifest["parameters"]["lookback"])
    return {"ticker": "HURN", "target_session": target, "price_basis": "as_stored",
            "bars": [{"session": day.date().isoformat(), "open": 100, "high": 102,
                      "low": 98, "close": 101, "volume": 1000} for day in dates]}


class WorkerTests(unittest.TestCase):
    def test_holiday_calendar_and_explicit_amount_proxy(self):
        model = worker()
        result = model.predict(request(model))
        self.assertEqual(result["context_end"], "2025-01-17")
        self.assertEqual(result["target_session"], "2025-01-21")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["reconstructed"])
        passed = model.predictor.calls[0]
        self.assertEqual(len(passed["df"]), 128)
        self.assertTrue((passed["df"]["amount"] == 100250).all())
        self.assertEqual(passed["pred_len"], 1)

    def test_missing_exchange_session_is_rejected_before_prediction(self):
        model = worker()
        sample = request(model)
        # Insert a non-session date without changing length or ascending order.
        for i, bar in enumerate(sample["bars"][:-1]):
            if pd.Timestamp(bar["session"]).weekday() == 4:
                sample["bars"][i]["session"] = (pd.Timestamp(bar["session"]) + pd.Timedelta(days=1)).date().isoformat()
                break
        with self.assertRaisesRegex(ValueError, "consecutive XNYS"):
            model.predict(sample)
        self.assertEqual(model.predictor.calls, [])

    def test_non_next_target_is_rejected(self):
        model = worker()
        sample = request(model)
        sample["target_session"] = "2025-01-22"
        with self.assertRaisesRegex(ValueError, "next XNYS"):
            model.predict(sample)
        self.assertEqual(model.predictor.calls, [])

    def test_forecast_incoherence_is_preserved_not_repaired(self):
        model = worker()
        model.predictor.predict = lambda **_: pd.DataFrame([dict(open=100, high=99, low=97, close=101, volume=1000, amount=100000)])
        result = model.predict(request(model))
        self.assertEqual(result["status"], "invalid_forecast")
        self.assertEqual(result["forecast"]["high"], 99)
        self.assertIn("incoherent", result["validation_error"])

    def test_later_actuals_do_not_affect_forecast_request_or_key(self):
        model = worker()
        sample = request(model)
        before = model.predict(sample)
        sample = {**deepcopy(sample), "actual_target_open": 50, "actual_target_close": 10000}
        after = model.predict(sample)
        self.assertEqual(before["request_key"], after["request_key"])
        pd.testing.assert_frame_equal(model.predictor.calls[0]["df"], model.predictor.calls[1]["df"])

    def test_completion_crossing_open_is_marked_reconstructed(self):
        model = worker()
        started = datetime(2025, 1, 21, 14, 29, 59, tzinfo=timezone.utc)
        completed = datetime(2025, 1, 21, 14, 30, 1, tzinfo=timezone.utc)
        with patch("tools.kronos.worker.utc_now", side_effect=[started, completed]):
            result = model.predict(request(model))
        self.assertTrue(result["reconstructed"])
        self.assertEqual(result["generated_at"], completed.isoformat())
        self.assertNotIn("context_available_at", result)
        self.assertEqual(result["context_not_before"], "2025-01-17T21:15:00+00:00")

    def test_derived_amount_numeric_overflow_is_rejected_before_inference(self):
        model = worker()
        sample = request(model)
        sample["bars"][-1]["volume"] = 1e38
        with self.assertRaisesRegex(ValueError, "amount proxy"):
            model.predict(sample)
        self.assertEqual(model.predictor.calls, [])


if __name__ == "__main__":
    unittest.main()
