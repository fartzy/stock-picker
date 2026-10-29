from stock_picker.storage.paper_book_store import PaperBookStore, PaperPick
import sqlite3

import pytest


def test_replace_all_round_trips(tmp_path):
    store = PaperBookStore(data_dir=tmp_path)
    store.replace_all(
        [
            PaperPick(
                as_of="2026-09-17",
                kind="fit",
                rank=1,
                ticker="TDTH",
                predicted=0.17,
                open_price=1.56,
                close_price=1.83,
                session_return=0.173,
            )
        ]
    )

    loaded = store.read()
    assert len(loaded) == 1
    assert loaded[0].ticker == "TDTH"
    assert loaded[0].session_return == 0.173


def test_replace_all_clears_previous_rows(tmp_path):
    store = PaperBookStore(data_dir=tmp_path)
    store.replace_all(
        [PaperPick("2026-09-16", "fit", 1, "XNDU", None, None, None, None)]
    )
    store.replace_all(
        [PaperPick("2026-09-17", "fit", 1, "TDTH", None, None, None, None)]
    )

    loaded = store.read()
    assert [p.ticker for p in loaded] == ["TDTH"]


def test_legacy_database_migration_preserves_rows_and_news_evidence(tmp_path):
    from stock_picker.storage.paper_book_store import _SCHEMA

    with sqlite3.connect(tmp_path / "paper_book.db") as conn:
        conn.executescript(_SCHEMA.replace("    news_check TEXT,\n", ""))
        conn.execute("INSERT INTO paper_picks (as_of, kind, rank, ticker) VALUES ('2026-09-30', 'rank', 1, 'TRLV')")
    store = PaperBookStore(data_dir=tmp_path)
    assert store.read()[0].ticker == "TRLV"
    assert store.read()[0].news_check is None
    evidence = {"status": "degraded", "article_count": 3, "issues": ["llm_unavailable"]}
    pick = store.read()[0]
    pick.news_check = evidence
    store.replace_all([pick])
    replay = PaperPick("2026-09-30", "fit", 1, "TRLV", None, None, None, None,
                       scan_id="replay", news_check={"status": "not_checked"})
    store.replace_scan("2026-09-30", "replay", [replay])
    # Reopening also runs migration again; both live/replay evidence survive.
    rows = PaperBookStore(data_dir=tmp_path).read()
    assert rows[0].news_check == evidence
    assert rows[1].news_check == {"status": "not_checked"}


def test_legacy_primary_key_allows_replays_without_replacing_live_history(tmp_path):
    from stock_picker.storage.paper_book_store import _SCHEMA

    legacy_schema = _SCHEMA.replace(
        "PRIMARY KEY (as_of, scan_id, kind, rank)", "PRIMARY KEY (as_of, kind, rank)"
    )
    with sqlite3.connect(tmp_path / "paper_book.db") as conn:
        conn.executescript(legacy_schema)
        conn.execute("CREATE INDEX paper_ticker ON paper_picks(ticker)")
        conn.execute(
            "INSERT INTO paper_picks (as_of, kind, rank, ticker, open_price, close_price, "
            "news_check, model_run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("2026-09-30", "rank", 1, "TRLV", 10.85, 10.90, '{"status":"reviewed"}', "original-model"),
        )
    store = PaperBookStore(data_dir=tmp_path)
    original = store.read()
    replay = PaperPick("2026-09-30", "rank", 1, "FICO", 0.02, 600, 610, 1 / 60,
                       scan_id="replay-one", model_run_id="archived-model")
    store.replace_scan(replay.as_of, replay.scan_id, [replay])
    second = PaperPick("2026-09-30", "rank", 1, "LAD", 0.01, 280, 285, 1 / 56,
                      scan_id="replay-two", model_run_id="another-model")
    store.replace_scan(second.as_of, second.scan_id, [second])
    store.replace_all(original)
    assert PaperBookStore(data_dir=tmp_path).read() == [*original, replay, second]
    assert original[0].news_check == {"status": "reviewed"}
    assert original[0].model_run_id == "original-model"
    with sqlite3.connect(tmp_path / "paper_book.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='paper_ticker'").fetchone()


def test_original_schema_upgrade_is_atomic(tmp_path):
    with sqlite3.connect(tmp_path / "paper_book.db") as conn:
        conn.execute("CREATE TABLE paper_picks (as_of TEXT, kind TEXT, rank INTEGER, ticker TEXT, "
                     "predicted REAL, open_price REAL, close_price REAL, session_return REAL, "
                     "PRIMARY KEY (as_of, kind, rank))")
        conn.execute("INSERT INTO paper_picks (as_of, kind, rank, ticker) VALUES ('2026-09-18', 'fit', 1, 'XENE')")

    class InterruptedUpgrade(PaperBookStore):
        def _migrate(self, conn):
            super()._migrate(conn)
            raise RuntimeError("interrupted upgrade")

    with pytest.raises(RuntimeError, match="interrupted upgrade"):
        InterruptedUpgrade(data_dir=tmp_path)
    with sqlite3.connect(tmp_path / "paper_book.db") as conn:
        assert len(list(conn.execute("PRAGMA table_info(paper_picks)"))) == 8
        assert conn.execute("SELECT ticker FROM paper_picks").fetchall() == [("XENE",)]
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='paper_picks_migrated'").fetchone() is None

    store = PaperBookStore(data_dir=tmp_path)
    replay = PaperPick("2026-09-18", "fit", 1, "LAD", None, None, None, None, scan_id="replay")
    store.replace_scan(replay.as_of, replay.scan_id, [replay])
    assert [pick.ticker for pick in store.read()] == ["XENE", "LAD"]
