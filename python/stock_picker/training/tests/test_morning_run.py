from types import SimpleNamespace

from stock_picker.training import morning as m


def test_second_lock_fails_while_first_is_held(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")

    first = m._try_lock_morning()
    assert first is not None
    assert m._try_lock_morning() is None
    first.close()
    third = m._try_lock_morning()
    assert third is not None
    third.close()


def test_run_morning_skips_when_8_32_job_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    monkeypatch.setattr(
        m,
        "pipeline_freshness",
        lambda: SimpleNamespace(ready_for_inference=True, detail="ready", as_of="2026-09-21"),
    )
    monkeypatch.setattr(
        m,
        "TrainingConfigStore",
        lambda: SimpleNamespace(read=lambda: SimpleNamespace(morning_job_enabled=False)),
    )
    quotes_called = []
    monkeypatch.setattr(m, "fetch_ticker_quotes", lambda *a, **k: quotes_called.append(True) or {})

    assert m.run_morning() == 0
    assert quotes_called == []


def test_run_morning_skips_when_a_scan_is_already_running(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    monkeypatch.setattr(
        m,
        "pipeline_freshness",
        lambda: SimpleNamespace(ready_for_inference=True, detail="ready", as_of="2026-09-21"),
    )
    monkeypatch.setattr(
        m,
        "TrainingConfigStore",
        lambda: SimpleNamespace(read=lambda: SimpleNamespace(morning_job_enabled=True)),
    )
    quotes_called = []
    monkeypatch.setattr(m, "fetch_ticker_quotes", lambda *a, **k: quotes_called.append(True) or {})
    held = m._try_lock_morning()
    assert held is not None

    assert m.run_morning(ignore_disabled=True) == 0
    assert quotes_called == []
    held.close()



def test_run_morning_ingests_universe_news_after_releasing_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    monkeypatch.setattr(
        m,
        "pipeline_freshness",
        lambda: SimpleNamespace(ready_for_inference=True, detail="ready", as_of="2026-09-23"),
    )
    monkeypatch.setattr(
        m,
        "TrainingConfigStore",
        lambda: SimpleNamespace(read=lambda: SimpleNamespace(morning_job_enabled=True)),
    )
    monkeypatch.setattr(m, "UniverseStore", lambda: SimpleNamespace(active_tickers=lambda: ["AAPL"]))
    monkeypatch.setattr(m, "fetch_ticker_quotes", lambda *a, **k: {"AAPL": {"open": 1}})
    monkeypatch.setattr(
        m,
        "score_from_quotes",
        lambda *a, **k: (
            SimpleNamespace(),
            SimpleNamespace(signals=[]),
            SimpleNamespace(as_of="2026-09-23", scored_count=1, signals=[]),
            0,
            "",
        ),
    )
    monkeypatch.setattr(m, "_payload_from", lambda *a, **k: {"as_of": "2026-09-23"})
    monkeypatch.setattr(m, "format_picks_email", lambda *a, **k: ("s", "b"))
    monkeypatch.setattr(m, "send_email", lambda *a, **k: True)
    ingest_saw_lock_free = []

    def _ingest():
        handle = m._try_lock_morning()
        ingest_saw_lock_free.append(handle is not None)
        if handle is not None:
            handle.close()

    monkeypatch.setattr(m, "ingest_universe_news", _ingest)

    assert m.run_morning(ignore_disabled=True) == 0
    assert ingest_saw_lock_free == [True]


def test_run_morning_still_succeeds_if_universe_news_ingest_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    monkeypatch.setattr(
        m,
        "pipeline_freshness",
        lambda: SimpleNamespace(ready_for_inference=True, detail="ready", as_of="2026-09-23"),
    )
    monkeypatch.setattr(
        m,
        "TrainingConfigStore",
        lambda: SimpleNamespace(read=lambda: SimpleNamespace(morning_job_enabled=True)),
    )
    monkeypatch.setattr(m, "UniverseStore", lambda: SimpleNamespace(active_tickers=lambda: ["AAPL"]))
    monkeypatch.setattr(m, "fetch_ticker_quotes", lambda *a, **k: {"AAPL": {"open": 1}})
    monkeypatch.setattr(
        m,
        "score_from_quotes",
        lambda *a, **k: (
            SimpleNamespace(),
            SimpleNamespace(signals=[]),
            SimpleNamespace(as_of="2026-09-23", scored_count=1, signals=[]),
            0,
            "",
        ),
    )
    monkeypatch.setattr(m, "_payload_from", lambda *a, **k: {"as_of": "2026-09-23"})
    monkeypatch.setattr(m, "format_picks_email", lambda *a, **k: ("s", "b"))
    monkeypatch.setattr(m, "send_email", lambda *a, **k: True)

    def _boom():
        raise RuntimeError("finnhub down")

    monkeypatch.setattr(m, "ingest_universe_news", _boom)

    assert m.run_morning(ignore_disabled=True) == 0
