from stock_picker.storage.paper_book_store import PaperBookStore, PaperPick


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
