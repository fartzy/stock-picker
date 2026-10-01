from types import SimpleNamespace
from datetime import date

from stock_picker.training import morning as m
from stock_picker.news_check import NewsCheck
from stock_picker.training.buy_signal import BuySignal, BuySignalResult


def test_second_lock_fails_while_first_is_held(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")

    first = m._try_lock_morning()
    assert first is not None
    assert m._try_lock_morning() is None
    assert m.morning_lock_held() is True
    first.close()
    assert m.morning_lock_held() is False
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
    waited = []
    monkeypatch.setattr(m, "wait_for_morning_snapshot", lambda: waited.append(True))
    monkeypatch.setattr(m, "fetch_morning_quotes", lambda *a, **k: quotes_called.append(True) or {})

    assert m.run_morning() == 0
    assert waited == [True]
    assert quotes_called == []


def test_run_morning_skips_scheduled_run_when_today_already_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "morning.lock")
    monkeypatch.setattr(
        m,
        "pipeline_freshness",
        lambda: SimpleNamespace(ready_for_inference=True, detail="ready", as_of="2026-09-25"),
    )
    monkeypatch.setattr(
        m,
        "TrainingConfigStore",
        lambda: SimpleNamespace(read=lambda: SimpleNamespace(morning_job_enabled=True)),
    )
    monkeypatch.setattr(m, "load_cached_signals", lambda kind="fit": {"as_of": "2026-09-25"} if kind == "rank" else None)
    quotes_called = []
    monkeypatch.setattr(m, "fetch_morning_quotes", lambda *a, **k: quotes_called.append(True) or {})

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
    monkeypatch.setattr(m, "load_cached_signals", lambda **k: None)
    quotes_called = []
    monkeypatch.setattr(m, "fetch_morning_quotes", lambda *a, **k: quotes_called.append(True) or {})
    held = m._try_lock_morning()
    assert held is not None

    assert m.run_morning(ignore_disabled=True) == 0
    assert quotes_called == []
    held.close()


def test_later_successful_news_batches_are_saved_even_without_flags(monkeypatch):
    freshness = SimpleNamespace(feature_snapshot_date="2026-09-29", model_trained_through="2026-09-29",
                                ready_for_inference=True, detail="ready")
    monkeypatch.setattr(m, "pipeline_freshness", lambda: freshness)
    monkeypatch.setattr("stock_picker.storage.ticker_blacklist_store.blacklisted_tickers", lambda: set())
    monkeypatch.setattr(m, "UniverseStore", lambda: SimpleNamespace(active_tickers=lambda: []))
    monkeypatch.setattr(m, "TrainingConfigStore", lambda: SimpleNamespace(read=lambda: SimpleNamespace(selected_run_id=None)))
    monkeypatch.setattr(m, "ModelStore", lambda: SimpleNamespace(exists=lambda name: False))
    monkeypatch.setattr(m, "FeatureStore", lambda: None)
    monkeypatch.setattr(m, "PriceStore", lambda: None)
    monkeypatch.setattr(m, "prepare_live_rows", lambda **kwargs: [])
    def result(n):
        return BuySignalResult("2026-09-30", 0.005, [BuySignal(f"T{i}", 0.01, 10, "2026-09-29") for i in range(n)])
    monkeypatch.setattr(m, "compute_rank_signals", lambda **kwargs: result(12))
    monkeypatch.setattr(m, "compute_buy_signals", lambda **kwargs: result(3))
    saved = []
    monkeypatch.setattr(m, "_write_signals", lambda payload, *args: saved.append(payload))
    monkeypatch.setattr(m, "write_picks_files", lambda *args: None)
    monkeypatch.setattr(m, "publish_picks", lambda *args: None)
    m.score_from_quotes({}, as_of=date(2026, 9, 30), earnings_fetcher=lambda *args: set(),
                        news_fetcher=lambda names, day: {name: NewsCheck(status="no_news", article_count=0, reviewed_count=0) for name in names})
    ranks = [payload for payload in saved if payload.get("kind") == "rank"]
    assert len(ranks) == 4  # Initial publish, then all three news batches.
    assert all(row["news_checked"] for row in ranks[-1]["signals"])
    assert all(row["news_check"]["status"] == "no_news" for row in ranks[-1]["signals"])



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
    monkeypatch.setattr(m, "fetch_morning_quotes", lambda *a, **k: {"AAPL": {"open": 1}})
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
    monkeypatch.setattr(m, "fetch_morning_quotes", lambda *a, **k: {"AAPL": {"open": 1}})
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
