"""First touch reads minute order, ignores extended hours, and caches each session once."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from stock_picker.paper import first_touch as ft

ET = ZoneInfo("America/New_York")


def bar(hh, mm, high, low):
    stamp = datetime(2026, 10, 15, hh, mm, tzinfo=ET).timestamp() * 1000
    return {"t": int(stamp), "o": low, "h": high, "l": low, "c": high, "v": 1}


def test_order_decides_which_level_was_first():
    bars = ft.regular_session_bars([bar(9, 30, 10.0, 9.9), bar(9, 31, 10.2, 10.0), bar(10, 0, 10.0, 9.7)])
    assert ft.first_touch(bars, target=10.1, stop=9.8) == "target"
    assert ft.first_touch(ft.regular_session_bars([bar(9, 30, 10.0, 9.7), bar(9, 31, 10.2, 10.0)]), 10.1, 9.8) == "stop"


def test_same_minute_neither_and_missing():
    assert ft.first_touch(ft.regular_session_bars([bar(9, 30, 10.2, 9.7)]), 10.1, 9.8) == "same_minute"
    assert ft.first_touch(ft.regular_session_bars([bar(9, 30, 10.0, 9.9)]), 10.1, 9.8) == "neither"
    assert ft.first_touch([], 10.1, 9.8) == "unavailable"


def test_premarket_prints_do_not_count():
    bars = ft.regular_session_bars([bar(8, 0, 10.5, 9.0), bar(9, 30, 10.2, 10.0)])
    assert len(bars) == 1
    assert ft.first_touch(bars, 10.1, 9.8) == "target"


def test_cache_fetches_each_session_once_and_skips_empty(tmp_path):
    calls = []

    def fetch(ticker, session):
        calls.append((ticker, session))
        return [bar(9, 30, 10.2, 10.0)] if ticker == "AAA" else []

    cache = ft.MinuteBarCache(fetch, cache_dir=tmp_path)
    day = date(2026, 10, 15)
    assert cache.bars("AAA", day) == cache.bars("AAA", day)
    assert cache.bars("BBB", day) == [] and cache.bars("BBB", day) == []
    assert calls == [("AAA", day), ("BBB", day), ("BBB", day)]
