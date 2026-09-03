from unittest.mock import MagicMock, patch

from stock_picker.tickers.universe import (
    build_universe,
    fetch_sp500_constituents,
    top_n_by_market_cap,
)


def test_fetch_sp500_constituents_sends_a_user_agent_and_parses_tickers():
    # Wikipedia 403s a header-less request -- this locks in that fetch_sp500_constituents
    # sends a User-Agent, and that "." in a symbol (e.g. BRK.B) becomes "-" for yfinance.
    fake_html = """
    <table>
      <tr><th>Symbol</th><th>Security</th></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
    </table>
    """
    mock_response = MagicMock()
    mock_response.text = fake_html
    mock_response.raise_for_status.return_value = None

    with patch(
        "stock_picker.tickers.universe.requests.get", return_value=mock_response
    ) as mock_get:
        tickers = fetch_sp500_constituents()

    assert tickers == ["AAPL", "BRK-B"]
    assert "User-Agent" in mock_get.call_args.kwargs["headers"]


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
