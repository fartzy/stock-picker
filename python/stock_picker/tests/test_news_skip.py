from stock_picker.news_skip import always_skip_news, news_blocks_buy


def test_always_skip_insider_sell_phrases():
    assert always_skip_news("Fastly's CTO Sells Over 33,000 Shares") is True
    assert always_skip_news("Insider sold 10,000 shares") is True
    assert always_skip_news("PIPE dilution/offering") is False
    assert always_skip_news(None) is False


def test_gap_up_still_buys_unless_always_skip():
    assert news_blocks_buy("PIPE", 11.0, 10.0) is False
    assert news_blocks_buy("PIPE", 9.0, 10.0) is True
    assert news_blocks_buy("CTO sold shares", 11.0, 10.0) is True


def test_incomplete_review_holds_even_without_a_flag_or_gap():
    for status in ("degraded", "error", "not_checked", "unknown"):
        assert news_blocks_buy(None, 11.0, 10.0, status)
    assert not news_blocks_buy(None, 11.0, 10.0, "complete")
    assert not news_blocks_buy(None, 11.0, 10.0, "no_news")
    assert not news_blocks_buy(None, 11.0, 10.0)


def test_corporate_actions_and_clinical_readouts_do_not_require_a_gap_down():
    assert news_blocks_buy("corporate action: CTVA spin-off", 11.0, 10.0)
    assert news_blocks_buy("clinical readout: NKTR Phase 2b data", 11.0, None)
