"""The news-outcome set labels judged, closed, live picks and scores the flag's direction."""

from stock_picker.paper import news_outcomes as no
from stock_picker.storage.paper_book_store import PaperPick


def pick(ticker, ret, flag=None, judge="grok", **extra):
    base = dict(as_of="2026-10-16", kind="rank", rank=1, ticker=ticker, predicted=0.1,
                open_price=10.5, close_price=None, session_return=ret, prev_close=10.0,
                news_check={"judge": judge, "flag": flag, "articles": [{"headline": f"{ticker} news"}]})
    return PaperPick(**{**base, **extra})


def test_rows_keep_only_judged_closed_live_picks():
    rows = no.news_outcome_rows([
        pick("A", -0.08, flag="dilution"),
        pick("B", 0.01),
        pick("C", None),
        pick("D", 0.02, judge=None),
        pick("E", 0.02, scan_id="replay"),
    ], {("2026-10-16", "rank", "A"): {"news_skip": 0.7}})
    assert [row["ticker"] for row in rows] == ["A", "B"]
    a = rows[0]
    assert a["flagged"] and a["dropped"] and a["jev_news_skip"] == 0.7
    assert abs(a["gap"] - 0.05) < 1e-9
    assert a["headlines"] == ["A news"]
    assert rows[1]["dropped"] is False


def test_flag_right_rate_is_share_of_pairs_where_flagged_did_worse():
    assert no.flag_points_right([-0.05], [0.01, 0.02]) == 1.0
    assert no.flag_points_right([0.05], [0.01, 0.02]) == 0.0
    assert no.flag_points_right([0.01], [0.01]) == 0.5
    assert no.flag_points_right([], [0.01]) is None


def test_summary_splits_flagged_and_clear():
    rows = no.news_outcome_rows([
        pick("A", -0.08, flag="dilution"),
        pick("B", 0.04, flag="takeover"),
        pick("C", 0.01),
        pick("D", -0.01),
    ])
    summary = no.summarize_news_outcomes(rows)
    assert summary["days"] == 1
    assert summary["flagged"]["n"] == 2 and summary["clear"]["n"] == 2
    assert summary["flagged"]["drop_rate"] == 0.5
    assert summary["clear"]["drop_rate"] == 0.0
    assert summary["flag_right_rate"] == 0.5
