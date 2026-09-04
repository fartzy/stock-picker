from stock_picker.features.quotes import quote_summaries


def test_quote_summaries_computes_diff_and_pct():
    raw = {"HOOD": {"open": 120.0, "last": 121.88}}

    summaries = quote_summaries(raw)

    assert summaries == [
        {
            "ticker": "HOOD",
            "open": 120.0,
            "last": 121.88,
            "diff": 1.88,
            "diff_pct": 0.0157,
            "prev_close": None,
            "gap": None,
            "gap_pct": None,
        }
    ]


def test_quote_summaries_computes_gap_when_prev_close_available():
    raw = {"HOOD": {"open": 120.0, "last": 121.88, "prev_close": 118.0}}

    summaries = quote_summaries(raw)

    assert summaries[0]["prev_close"] == 118.0
    assert summaries[0]["gap"] == 2.0
    assert summaries[0]["gap_pct"] == round(2.0 / 118.0, 4)


def test_quote_summaries_omits_nothing_it_wasnt_given():
    # a ticker yfinance couldn't fetch is simply absent from the input dict --
    # quote_summaries must not crash trying to look it up.
    raw = {"HOOD": {"open": 120.0, "last": 121.88}}

    summaries = quote_summaries(raw)

    assert {s["ticker"] for s in summaries} == {"HOOD"}


def test_quote_summaries_handles_empty_input():
    assert quote_summaries({}) == []
