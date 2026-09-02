from stock_picker.tickers.universe import build_universe, top_n_by_market_cap


def test_top_n_by_market_cap_orders_descending():
    market_caps = {"AAPL": 3.0e12, "MSFT": 2.5e12, "GOOG": 1.8e12}
    assert top_n_by_market_cap(market_caps, n=2) == ["AAPL", "MSFT"]


def test_top_n_by_market_cap_respects_n():
    market_caps = {f"T{i}": float(i) for i in range(10)}
    assert len(top_n_by_market_cap(market_caps, n=3)) == 3


def test_build_universe_tags_source():
    market_caps = {"AAPL": 3.0e12, "MSFT": 2.5e12}
    universe = build_universe(market_caps, n=2, manual_additions=["ZZZ"])

    assert universe == {"AAPL": "market_cap", "MSFT": "market_cap", "ZZZ": "manual"}


def test_build_universe_market_cap_takes_precedence_on_overlap():
    market_caps = {"AAPL": 3.0e12}
    universe = build_universe(market_caps, n=1, manual_additions=["AAPL"])

    assert universe == {"AAPL": "market_cap"}
