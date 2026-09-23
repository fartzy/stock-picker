from datetime import date

import pandas as pd

from stock_picker.paper.book import paper_book_view, rebuild_paper_book
from stock_picker.storage.paper_book_store import PaperBookStore
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
