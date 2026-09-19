from datetime import date

import pandas as pd

from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.morning_check import fake_quotes_from_last_close


def test_fake_quotes_use_last_close_as_open(tmp_path):
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

    quotes = fake_quotes_from_last_close(universe=universe, prices=prices)

    assert len(quotes) == 1
    assert quotes[0].ticker == "AAPL"
    assert quotes[0].fake_open == 110.0
    assert quotes[0].last_close == 110.0


def test_both_runs_rank_and_fit_and_reports_a_wall_clock(monkeypatch, tmp_path):
    from stock_picker.training import morning_check as mc
    from stock_picker.training.buy_signal import BuySignal, BuySignalResult

    def fake_rank(**kwargs):
        return BuySignalResult(as_of="2026-09-19", threshold=0.0, scored_count=2, signals=[])

    def fake_fit(**kwargs):
        return BuySignalResult(as_of="2026-09-19", threshold=0.005, scored_count=2, signals=[])

    monkeypatch.setattr(mc, "compute_rank_signals", fake_rank)
    monkeypatch.setattr(mc, "compute_buy_signals", fake_fit)
    monkeypatch.setattr(
        mc,
        "fake_quotes_from_last_close",
        lambda universe=None, prices=None: [mc.FakeQuote("AAPL", 110.0, 110.0)],
    )

    result = mc.run_morning_check(which="both")

    names = [p.which for p in result.passes]
    assert names == ["rank", "fit", "both"]
    both = result.passes[-1]
    rank_s = result.passes[0].seconds
    fit_s = result.passes[1].seconds
    assert both.seconds <= rank_s + fit_s + 0.5
