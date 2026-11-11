from datetime import datetime, timedelta
import math
from zoneinfo import ZoneInfo

import pytest

from stock_picker.features.intraday_context import build_intraday_context

ET = ZoneInfo("America/New_York")
OPEN = datetime(2026, 11, 10, 9, 30, tzinfo=ET)


def bars(start, volume=10, price=100):
    return [dict(t=int((start + timedelta(minutes=i)).timestamp()*1000),
                 o=price, c=price, vw=price, v=volume) for i in range(60)]


def inputs():
    history = []
    for i in range(1, 21):
        start = OPEN - timedelta(days=i)
        history.append((start, start + timedelta(hours=6, minutes=30), bars(start)))
    return dict(stock_bars=bars(OPEN, volume=20), spy_bars=bars(OPEN), history=history,
                observed_at=OPEN+timedelta(minutes=30), session_open_at=OPEN,
                stock_return_from_open=-.04, stock_return_15m=-.02)


def test_known_market_and_same_clock_ratios():
    values, coverage = build_intraday_context(**inputs())
    assert values["stock_minus_spy_from_open"] == pytest.approx(-.04)
    assert values["spy_return_15m"] == 0
    assert values["log_same_clock_dollar_volume"] == pytest.approx(math.log(2))
    assert values["log_same_clock_5m_dollar_volume"] == pytest.approx(math.log(2))
    assert coverage["cumulative_reference_sessions"] == 20


def test_future_or_unfinished_bars_cannot_change_features():
    data = inputs()
    expected = build_intraday_context(**data)
    cutoff = data["observed_at"].timestamp()*1000
    for field in ("stock_bars", "spy_bars"):
        for bar in data[field]:
            if bar["t"] >= cutoff:
                bar.update(c=9000, vw=9000, v=1000000)
    assert build_intraday_context(**data) == expected


def test_future_history_duplicate_history_and_naive_times_rejected():
    data = inputs()
    data["history"][0] = (OPEN, OPEN+timedelta(hours=6), bars(OPEN))
    with pytest.raises(ValueError, match="prior"):
        build_intraday_context(**data)
    data = inputs(); data["history"][0] = data["history"][1]
    with pytest.raises(ValueError, match="unique"):
        build_intraday_context(**data)
    data = inputs(); data["observed_at"] = data["observed_at"].replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        build_intraday_context(**data)


def test_missing_history_and_early_closes_are_unknown_not_zero():
    data = inputs(); data["history"] = data["history"][:9]
    assert math.isnan(build_intraday_context(**data)[0]["log_same_clock_dollar_volume"])
    data = inputs(); data["history"] = [(start, start+timedelta(minutes=20), rows)
                                      for start, _, rows in data["history"]]
    result, coverage = build_intraday_context(**data)
    assert coverage["cumulative_reference_sessions"] == 0
    assert math.isnan(result["log_same_clock_dollar_volume"])


def test_split_share_count_alone_does_not_change_dollar_volume():
    data = inputs(); before = build_intraday_context(**data)
    for row in data["stock_bars"]:
        row["v"] *= 10; row["vw"] /= 10
    assert build_intraday_context(**data) == before


def test_sparse_or_duplicate_current_bars_not_silently_complete():
    data = inputs(); data["stock_bars"] = data["stock_bars"][::2]
    assert math.isnan(build_intraday_context(**data)[0]["log_same_clock_dollar_volume"])
    data = inputs(); data["spy_bars"].append(data["spy_bars"][0])
    with pytest.raises(ValueError, match="duplicate"):
        build_intraday_context(**data)
