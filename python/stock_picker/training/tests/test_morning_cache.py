import json

from stock_picker.training.morning import load_cached_signals


def test_load_cached_signals_returns_today_payload(tmp_path):
    payload = {"as_of": "2026-09-11", "signals": [{"ticker": "FLY", "predicted_return": 0.018}]}
    (tmp_path / "2026-09-11.json").write_text(json.dumps(payload))
    (tmp_path / "latest.json").write_text(json.dumps(payload))

    loaded = load_cached_signals(as_of="2026-09-11", signal_dir=tmp_path)

    assert loaded["signals"][0]["ticker"] == "FLY"


def test_load_cached_signals_ignores_yesterdays_latest(tmp_path):
    payload = {"as_of": "2026-09-10", "signals": [{"ticker": "OLD", "predicted_return": 0.01}]}
    (tmp_path / "latest.json").write_text(json.dumps(payload))

    assert load_cached_signals(as_of="2026-09-11", signal_dir=tmp_path) is None


def test_load_cached_signals_is_none_when_the_job_has_not_run(tmp_path):
    assert load_cached_signals(signal_dir=tmp_path) is None


def test_write_signals_lands_under_the_app_data_dir(tmp_path):
    from stock_picker.training.morning import _write_signals

    payload = {"as_of": "2026-09-11", "signals": []}
    dated = _write_signals(payload, "2026-09-11", signal_dir=tmp_path)

    assert dated == tmp_path / "2026-09-11.json"
    assert (tmp_path / "latest.json").is_file()
    assert json.loads((tmp_path / "latest.json").read_text())["as_of"] == "2026-09-11"
