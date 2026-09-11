from stock_picker.training.notify import (
    format_not_ready_email,
    format_picks_email,
    notify_address,
    write_picks_files,
)


def test_notify_address_reads_a_bare_email_file(tmp_path, monkeypatch):
    monkeypatch.delenv("STOCK_PICKER_NOTIFY_EMAIL", raising=False)
    key_file = tmp_path / "notify-email.txt"
    key_file.write_text("michaeleartz@gmail.com\n")

    assert notify_address(key_file=key_file) == "michaeleartz@gmail.com"


def test_notify_address_env_wins(tmp_path, monkeypatch):
    key_file = tmp_path / "notify-email.txt"
    key_file.write_text("from-file@example.com\n")
    monkeypatch.setenv("STOCK_PICKER_NOTIFY_EMAIL", "from-env@example.com")

    assert notify_address(key_file=key_file) == "from-env@example.com"


def test_format_picks_email_includes_top_rows_and_earnings_skip():
    payload = {
        "as_of": "2026-09-11",
        "scored_count": 1992,
        "signals": [
            {"ticker": "FLY", "predicted_return": 0.0182, "open_price": 21.05},
            {"ticker": "IREN", "predicted_return": 0.0180, "open_price": 44.62},
        ],
        "skipped": [{"ticker": "ORCL", "reason": "earnings on or since the prior session"}],
        "freshness": {"detail": "ready for this morning's opens."},
    }

    subject, body = format_picks_email(payload)

    assert subject == "stockpicker 2026-09-11: 2 picks"
    assert "FLY" in body and "1.82%" in body
    assert "Skipped earnings: ORCL" in body


def test_format_not_ready_email():
    subject, body = format_not_ready_email("2026-09-11", "model trained through Monday")

    assert "not ready" in subject
    assert "model trained through Monday" in body


def test_write_picks_files_partitions_by_date_and_updates_latest(tmp_path):
    body = "FLY  1.82%\n"
    dated = write_picks_files("2026-09-11", body, root=tmp_path)

    assert dated == tmp_path / "picks" / "2026" / "09" / "11.txt"
    assert dated.read_text() == body
    assert (tmp_path / "picks" / "latest.txt").read_text() == body