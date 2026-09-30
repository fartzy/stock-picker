from stock_picker.storage.paper_book_store import PaperBookStore, PaperPick
import sqlite3


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
