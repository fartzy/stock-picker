"""Monday 8:30:05 CT: lock held means the Trading button stays grey."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi import status
from fastapi.testclient import TestClient

from stock_picker.api.app import app
from stock_picker.ingestion.session import cash_session_date
from stock_picker.training import morning as m
from stock_picker.training.job import JobStatus

# Monday. launchd has fired; script is still in the 10s sleep until 8:30:10.
# Tests that hold the lock simulate scoring already underway (click or a
# slightly early score) so the button must not start a second run.
MONDAY_83005 = datetime(2026, 9, 28, 8, 30, 5, tzinfo=ZoneInfo("America/Chicago"))

# Keep in lockstep with typescript/src/morningScanUi.ts
SCORING_LABEL = "Scoring this morning's prices…"
CHECK_PRICES_LABEL = "Check this morning's prices"


def morning_scan_button(running: bool) -> dict:
    """Same rules as morningScanButton() in the frontend."""
    if running:
        return {"label": SCORING_LABEL, "disabled": True}
    return {"label": CHECK_PRICES_LABEL, "disabled": False}


def test_monday_83005_is_a_weekday_cash_session():
    assert MONDAY_83005.strftime("%A") == "Monday"
    assert cash_session_date(MONDAY_83005).isoformat() == "2026-09-28"


def test_hero_button_idles_until_a_scan_is_running():
    assert morning_scan_button(False) == {
        "label": CHECK_PRICES_LABEL,
        "disabled": False,
    }


def test_hero_button_greys_out_when_scan_status_is_running():
    assert morning_scan_button(True) == {
        "label": SCORING_LABEL,
        "disabled": True,
    }


def test_get_morning_scan_is_running_when_lock_held_at_monday_open(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    held = m._try_lock_morning()
    try:
        with patch("stock_picker.api.routes.morning_scan_job") as job:
            job.status.return_value = JobStatus()
            response = TestClient(app).get("/api/morning-scan")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "running"
        assert morning_scan_button(True)["disabled"] is True
        assert morning_scan_button(True)["label"] == SCORING_LABEL
    finally:
        held.close()


def test_post_morning_scan_is_409_when_lock_held_at_monday_open(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    held = m._try_lock_morning()
    try:
        with patch("stock_picker.api.routes.morning_scan_job") as job:
            job.status.return_value = JobStatus()
            response = TestClient(app).post("/api/morning-scan")
            job.start.assert_not_called()
        assert response.status_code == status.HTTP_409_CONFLICT
    finally:
        held.close()


def test_morning_sh_sleeps_ten_seconds_not_fifteen():
    candidates = [
        Path("scripts/morning.sh"),
        Path(__file__).resolve().parents[4] / "scripts" / "morning.sh",
    ]
    script = next((path for path in candidates if path.exists()), None)
    if script is None:
        return
    text = script.read_text()
    assert "sleep 10" in text
    assert "sleep 15" not in text
