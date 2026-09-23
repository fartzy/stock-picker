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
