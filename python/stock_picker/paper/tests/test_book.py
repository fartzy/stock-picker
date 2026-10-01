from datetime import date

import pandas as pd

from stock_picker.paper.book import load_paper_book, paper_book_view, rebuild_paper_book
from stock_picker.storage.paper_book_store import PaperBookStore, PaperPick
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.scan_store import ScanStore


def test_rebuild_stores_every_name_not_just_top_five(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-17",
        "fit",
        {
            "as_of": "2026-09-17",
            "kind": "fit",
            "signals": [{"ticker": f"T{i}", "predicted_return": 0.01} for i in range(8)],
        },
    )
    book = PaperBookStore(data_dir=tmp_path / "paper")

    picks = rebuild_paper_book(
        completed_through=date(2026, 9, 16),
        scan_store=scans,
        price_store=PriceStore(data_dir=tmp_path / "prices"),
        paper_store=book,
        quote_fetcher=lambda tickers, as_of: {},
        news_fetcher=lambda tickers, as_of: {},
    )

    assert len(picks) == 8
    sliced = paper_book_view(picks, kind="fit", top_k=5)
    assert sliced["n_picks"] == 5
    assert paper_book_view(picks, kind="rank")["n_picks"] == 0


def test_paper_book_view_slices_fit_and_rank_separately():
    from stock_picker.storage.paper_book_store import PaperPick

    picks = [
        PaperPick(
            as_of="2026-09-24",
            kind="fit",
            rank=i,
            ticker=f"F{i}",
            predicted=0.01,
            open_price=10.0,
            close_price=10.1,
            session_return=0.01,
        )
        for i in range(1, 6)
    ] + [
        PaperPick(
            as_of="2026-09-24",
            kind="rank",
            rank=i,
            ticker=f"R{i}",
            predicted=0.5,
            open_price=10.0,
            close_price=10.1,
            session_return=0.01,
        )
        for i in range(1, 6)
    ]
    sliced = paper_book_view(picks, kind="both", fit_top_k=2, rank_top_k=4)
    assert [row["ticker"] for row in sliced["days"][0]["fit"]] == ["F1", "F2"]
    assert [row["ticker"] for row in sliced["days"][0]["rank"]] == ["R1", "R2", "R3", "R4"]


def test_degraded_saved_review_is_held_out_of_what_if_returns():
    picks = [
        PaperPick(as_of="2026-10-02", kind="rank", rank=1, ticker="HOLD",
                  predicted=0.5, open_price=10.0, close_price=8.0,
                  session_return=-0.2, news_checked=0,
                  news_check={"status": "degraded", "issues": ["llm_unavailable"]}),
        PaperPick(as_of="2026-10-02", kind="rank", rank=2, ticker="CLEAR",
                  predicted=0.3, open_price=10.0, close_price=11.0,
                  session_return=0.1, news_checked=1,
                  news_check={"status": "no_news"}),
    ]
    view = paper_book_view(picks, kind="rank")
    assert view["days"][0]["rank"][0]["news_blocks"] is True
    assert view["rank_stats"]["n_avoid"] == 1
    assert view["rank_stats"]["n_scored"] == 1


def test_rebuild_preserves_saved_degraded_review_status(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write("2026-10-02", "rank", {
        "as_of": "2026-10-02", "kind": "rank", "signals": [{
            "ticker": "HOLD", "predicted_return": 0.5, "open_price": 10.0,
            "news_checked": False, "news_check": {"status": "degraded"},
        }],
    })
    picks = rebuild_paper_book(
        completed_through=date(2026, 10, 1), scan_store=scans,
        price_store=PriceStore(data_dir=tmp_path / "prices"),
        paper_store=PaperBookStore(data_dir=tmp_path / "paper"),
        quote_fetcher=lambda tickers, as_of: {},
        news_fetcher=lambda tickers, as_of: {},
    )
    assert picks[0].news_checked == 0
    assert picks[0].news_check == {"status": "degraded"}


def test_load_paper_book_does_not_rebuild(tmp_path):
    book = PaperBookStore(data_dir=tmp_path / "paper")
    assert load_paper_book(paper_store=book) == []


def test_rebuild_one_completed_day_preserves_history_and_replays(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-30",
        "rank",
        {"as_of": "2026-09-30", "kind": "rank", "signals": [
            {"ticker": "WED", "predicted_return": 0.2, "open_price": 10.0, "news_checked": True}
        ]},
    )
    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(
        "WED",
        pd.DataFrame(
            {"Open": [10.0], "Close": [11.0]},
            index=pd.DatetimeIndex(["2026-09-30"]),
        ),
    )
    book = PaperBookStore(data_dir=tmp_path / "paper")
    older = PaperPick("2026-09-29", "rank", 1, "OLD", 0.1, 8.0, 9.0, 0.125)
    replay = PaperPick("2026-09-30", "fit", 1, "REPLAY", 0.1, 8.0, 9.0, 0.125, scan_id="replay-1")
    book.replace_all([older])
    book.replace_scan("2026-09-30", "replay-1", [replay])

    for _ in range(2):
        rows = rebuild_paper_book(
            completed_through=date(2026, 9, 30),
            scan_store=scans,
            price_store=prices,
            paper_store=book,
            quote_fetcher=lambda tickers, as_of: {},
            news_fetcher=lambda tickers, as_of: {},
            only_day="2026-09-30",
        )

    assert len(rows) == 1
    assert rows[0].ticker == "WED"
    assert rows[0].session_return == (11.0 / 10.0) - 1.0
    assert book.read() == [older, rows[0], replay]


def test_rebuild_one_day_without_saved_scan_does_not_delete_existing_rows(tmp_path):
    book = PaperBookStore(data_dir=tmp_path / "paper")
    existing = PaperPick("2026-09-30", "fit", 1, "KEEP", 0.1, 8.0, 9.0, 0.125)
    book.replace_all([existing])

    rows = rebuild_paper_book(
        completed_through=date(2026, 9, 30),
        scan_store=ScanStore(data_dir=tmp_path / "scans"),
        price_store=PriceStore(data_dir=tmp_path / "prices"),
        paper_store=book,
        only_day="2026-09-30",
    )

    assert rows == []
    assert book.read() == [existing]


def test_rebuild_scores_open_to_close(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-17",
        "fit",
        {
            "as_of": "2026-09-17",
            "kind": "fit",
            "signals": [
                {"ticker": "TDTH", "predicted_return": 0.17},
                {"ticker": "QH", "predicted_return": 0.01},
            ],
        },
    )
    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(
        "TDTH",
        pd.DataFrame(
            {"Open": [1.56], "High": [1.83], "Low": [1.50], "Close": [1.83], "Volume": [1.0]},
            index=pd.DatetimeIndex(["2026-09-17"]),
        ),
    )
    book = PaperBookStore(data_dir=tmp_path / "paper")

    picks = rebuild_paper_book(
        completed_through=date(2026, 9, 17),
        scan_store=scans,
        price_store=prices,
        paper_store=book,
        quote_fetcher=lambda tickers, as_of: {},
        news_fetcher=lambda tickers, as_of: {},
    )

    by_ticker = {p.ticker: p for p in picks}
    assert by_ticker["TDTH"].session_return == (1.83 / 1.56) - 1
    assert by_ticker["QH"].session_return is None


def test_session_quotes_from_prices_uses_the_daily_bar(tmp_path):
    from stock_picker.paper.book import session_quotes_from_prices

    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(
        "WRBY",
        pd.DataFrame(
            {
                "Open": [23.18, 26.27],
                "High": [26.95, 27.0],
                "Low": [23.0, 26.2],
                "Close": [26.27, 26.71],
                "Volume": [1.0, 1.0],
            },
            index=pd.DatetimeIndex(["2026-09-24", "2026-09-25"]),
        ),
    )
    quotes = session_quotes_from_prices(["WRBY"], date(2026, 9, 25), prices)
    assert quotes["WRBY"]["open"] == 26.27
    assert quotes["WRBY"]["prev_close"] == 26.27
    assert quotes["WRBY"]["last"] == 26.71


def test_rebuild_uses_scan_open_not_daily_bar(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-25",
        "rank",
        {
            "as_of": "2026-09-25",
            "kind": "rank",
            "signals": [{"ticker": "WRBY", "predicted_return": 0.99, "open_price": 23.18}],
        },
    )
    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(
        "WRBY",
        pd.DataFrame(
            {"Open": [26.27], "High": [27.0], "Low": [23.0], "Close": [26.71], "Volume": [1.0]},
            index=pd.DatetimeIndex(["2026-09-25"]),
        ),
    )
    picks = rebuild_paper_book(
        completed_through=date(2026, 9, 25),
        scan_store=scans,
        price_store=prices,
        paper_store=PaperBookStore(data_dir=tmp_path / "paper"),
        quote_fetcher=lambda tickers, as_of: {},
        news_fetcher=lambda tickers, as_of: {},
    )

    assert picks[0].open_price == 23.18
    assert picks[0].close_price == 26.71
    assert picks[0].session_return == (26.71 / 23.18) - 1


def test_live_quote_fills_a_missing_daily_bar(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-18",
        "rank",
        {
            "as_of": "2026-09-18",
            "kind": "rank",
            "signals": [{"ticker": "XENE", "predicted_return": 0.8}],
        },
    )
    book = PaperBookStore(data_dir=tmp_path / "paper")

    picks = rebuild_paper_book(
        completed_through=date(2026, 9, 18),
        scan_store=scans,
        price_store=PriceStore(data_dir=tmp_path / "prices"),
        paper_store=book,
        quote_fetcher=lambda tickers, as_of: {"XENE": {"open": 40.0, "last": 41.0}},
        news_fetcher=lambda tickers, as_of: {},
    )

    assert picks[0].open_price == 40.0
    assert picks[0].close_price == 41.0
    assert picks[0].session_return == (41.0 / 40.0) - 1
    view = paper_book_view(picks, kind="rank")
    assert view["rank_stats"]["wins"] == 1
    assert view["rank_stats"]["hit_rate"] == 1.0


def test_news_flag_from_scan_marks_avoid(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-18",
        "rank",
        {
            "as_of": "2026-09-18",
            "kind": "rank",
            "signals": [
                {
                    "ticker": "XENE",
                    "predicted_return": 0.8,
                    "news_flag": "Xenon pauses Phase 3 clinical trial",
                }
            ],
        },
    )
    book = PaperBookStore(data_dir=tmp_path / "paper")

    picks = rebuild_paper_book(
        completed_through=date(2026, 9, 17),
        scan_store=scans,
        price_store=PriceStore(data_dir=tmp_path / "prices"),
        paper_store=book,
        quote_fetcher=lambda tickers, as_of: {},
        news_fetcher=lambda tickers, as_of: {},
    )

    assert picks[0].news_flag == "Xenon pauses Phase 3 clinical trial"
    assert paper_book_view(picks, kind="rank")["rank_stats"]["n_avoid"] == 0


def test_news_plus_gap_down_is_avoid_gap_up_is_still_buy(tmp_path):
    scans = ScanStore(data_dir=tmp_path / "scans")
    scans.write(
        "2026-09-18",
        "rank",
        {
            "as_of": "2026-09-18",
            "kind": "rank",
            "signals": [
                {"ticker": "DOWN", "predicted_return": 0.8, "news_flag": "PIPE"},
                {"ticker": "UP", "predicted_return": 0.7, "news_flag": "PIPE"},
            ],
        },
    )
    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(
        "DOWN",
        pd.DataFrame(
            {"Open": [10.0, 9.0], "High": [10.0, 9.0], "Low": [10.0, 9.0], "Close": [10.0, 8.5], "Volume": [1.0, 1.0]},
            index=pd.DatetimeIndex(["2026-09-17", "2026-09-18"]),
        ),
    )
    prices.write(
        "UP",
        pd.DataFrame(
            {"Open": [10.0, 11.0], "High": [10.0, 11.0], "Low": [10.0, 11.0], "Close": [10.0, 11.5], "Volume": [1.0, 1.0]},
            index=pd.DatetimeIndex(["2026-09-17", "2026-09-18"]),
        ),
    )
    book = PaperBookStore(data_dir=tmp_path / "paper")
    picks = rebuild_paper_book(
        completed_through=date(2026, 9, 18),
        scan_store=scans,
        price_store=prices,
        paper_store=book,
        quote_fetcher=lambda tickers, as_of: {},
        news_fetcher=lambda tickers, as_of: {},
    )
    view = paper_book_view(picks, kind="rank")
    by_ticker = {row["ticker"]: row for row in view["days"][0]["rank"]}
    assert by_ticker["DOWN"]["news_blocks"] is True
    assert by_ticker["UP"]["news_blocks"] is False
    assert view["rank_stats"]["n_avoid"] == 1
    assert view["rank_stats"]["n_scored"] == 1
    assert view["rank_stats"]["avg"] == (11.5 / 11.0) - 1
    assert view["days"][0]["rank_avg"] == (11.5 / 11.0) - 1
