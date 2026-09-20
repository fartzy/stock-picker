from datetime import date
import random

import pandas as pd

from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.morning_check import randomized_fake_quotes


def test_fake_quotes_jitter_open_around_last_close(tmp_path):
    universe = UniverseStore(data_dir=tmp_path / "universe")
    universe.sync({"AAPL": "Technology"}, as_of=date(2026, 9, 1))
    prices = PriceStore(data_dir=tmp_path / "prices")
    prices.write(
        "AAPL",
        pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [110.0], "Volume": [1.0]},
            index=pd.DatetimeIndex(["2026-09-17"]),
        ),
    )

    quotes = randomized_fake_quotes(universe=universe, prices=prices, rng=random.Random(0))

    assert len(quotes) == 1
    assert quotes[0].ticker == "AAPL"
    assert quotes[0].last_close == 110.0
    assert quotes[0].fake_open != 110.0
    assert 110.0 * 0.97 <= quotes[0].fake_open <= 110.0 * 1.03


def test_both_uses_score_from_quotes(monkeypatch):
    from stock_picker.training import morning_check as mc
    from stock_picker.training.buy_signal import BuySignalResult

    called = {}

    def fake_score(quotes, threshold=0.005, persist=True, news_fetcher=None, earnings_fetcher=None):
        called["persist"] = persist
        called["news"] = news_fetcher is not None
        rank = BuySignalResult(as_of="2026-09-19", threshold=0.0, scored_count=2, signals=[])
        fit = BuySignalResult(as_of="2026-09-19", threshold=0.005, scored_count=2, signals=[])
        return None, rank, fit, 0, ""

    monkeypatch.setattr(mc, "score_from_quotes", fake_score)
    monkeypatch.setattr(
        mc,
        "randomized_fake_quotes",
        lambda universe=None, prices=None, rng=None: [mc.FakeQuote("AAPL", 111.0, 110.0)],
    )

    result = mc.run_morning_check(which="both", news_fetcher=lambda t, d: {})
    assert called["persist"] is False
    assert [p.which for p in result.passes] == ["rank", "fit", "both"]
    assert result.n_quotes == 1
