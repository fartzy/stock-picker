from datetime import date

import pandas as pd
import pytest

from stock_picker.features.tests.fixtures import synthetic_history
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.training.dataset import GAP_COLUMN
from stock_picker.training.live_rows import _index_dates, prepare_live_rows, prepare_one

_AS_OF = date(2026, 9, 9)
_SNAPSHOT = date(2026, 9, 8)


def _stores(tmp_path):
    return FeatureStore(data_dir=tmp_path / "features"), PriceStore(data_dir=tmp_path / "prices")


def _seed_features(feature_store, ticker, snapshot_date=_SNAPSHOT, extra=None):
    dates = [snapshot_date]
    values = [1.0]
    if extra:
        for day, value in extra:
            dates.append(day)
            values.append(value)
    frame = pd.DataFrame(
        {"some_feature": values, GAP_COLUMN: [0.0] * len(values)},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    feature_store.write(ticker, frame)


def _quote(open_price=101.0, prev_close=100.0):
    return {"open": open_price, "last": open_price + 1, "prev_close": prev_close}


def test_index_dates_strips_tz_to_calendar_dates():
    index = pd.DatetimeIndex(
        ["2026-09-08 16:00:00-04:00", "2026-09-09 09:30:00-04:00"],
        tz="America/New_York",
    )
    assert list(_index_dates(index)) == [date(2026, 9, 8), date(2026, 9, 9)]


def test_prepare_one_skips_blocked_without_reading_stores(tmp_path):
    feature_store, price_store = _stores(tmp_path)
    row = prepare_one(
        "TDTH",
        quotes={"TDTH": _quote()},
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        blocked={"TDTH"},
    )
    assert row.skipped == {"ticker": "TDTH", "reason": "blacklisted"}
    assert row.row is None


def test_prepare_one_skips_earnings(tmp_path):
    feature_store, price_store = _stores(tmp_path)
    row = prepare_one(
        "AAPL",
        quotes={"AAPL": _quote()},
        earnings={"AAPL"},
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        blocked=set(),
    )
    assert row.skipped["ticker"] == "AAPL"
    assert "earnings" in row.skipped["reason"]


def test_prepare_one_skips_missing_quote_and_prev_close(tmp_path):
    feature_store, price_store = _stores(tmp_path)
    no_quote = prepare_one(
        "AAPL",
        quotes={},
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        blocked=set(),
    )
    assert no_quote.skipped == {"ticker": "AAPL", "reason": "no live quote available"}

    no_prev = prepare_one(
        "AAPL",
        quotes={"AAPL": {"open": 101.0, "last": 102.0, "prev_close": None}},
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        blocked=set(),
    )
    assert no_prev.skipped == {"ticker": "AAPL", "reason": "no previous close available"}


def test_prepare_one_skips_when_every_feature_row_is_as_of_or_later(tmp_path):
    feature_store, price_store = _stores(tmp_path)
    _seed_features(feature_store, "AAPL", snapshot_date=_AS_OF)
    row = prepare_one(
        "AAPL",
        quotes={"AAPL": _quote()},
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        blocked=set(),
    )
    assert row.skipped == {
        "ticker": "AAPL",
        "reason": "no feature snapshot before this morning",
    }


def test_prepare_one_uses_the_last_snapshot_strictly_before_as_of(tmp_path):
    feature_store, price_store = _stores(tmp_path)
    _seed_features(
        feature_store,
        "AAPL",
        snapshot_date=_SNAPSHOT,
        extra=[(_AS_OF, 99.0)],
    )
    row = prepare_one(
        "AAPL",
        quotes={"AAPL": _quote(open_price=102.0, prev_close=100.0)},
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        blocked=set(),
    )
    assert row.skipped is None
    assert row.snapshot_date == _SNAPSHOT.isoformat()
    assert row.open_price == 102.0
    assert row.prev_close == 100.0
    assert row.row is not None
    assert row.row.iloc[0]["some_feature"] == pytest.approx(1.0)
    assert row.row.iloc[0][GAP_COLUMN] == pytest.approx(0.02)


_SENTINEL = 999.0


def test_prepare_one_excluded_features_skips_recompute_but_keeps_other_open_known_columns(tmp_path):
    # A currently-pruned open-known column must not get recomputed -- its
    # sentinel from last night's persisted snapshot should survive untouched
    # -- while a NOT-excluded open-known column must still be freshly
    # recomputed from today's open (see features/open_pattern_seasonality.py
    # and training/inference.py's build_inference_row).
    feature_store, price_store = _stores(tmp_path)
    history = synthetic_history(n=140)
    price_store.write("AAPL", history)
    snapshot_date = history.index[-1].date()
    # Next business day after the snapshot -- inside check_freshness's TTL
    # (see training/freshness.py), unlike this file's usual fixed _AS_OF,
    # which is unrelated to synthetic_history's own date range.
    as_of = (history.index[-1] + pd.tseries.offsets.BDay(1)).date()
    frame = pd.DataFrame(
        {
            "some_feature": [1.0],
            GAP_COLUMN: [0.0],
            "seq2_open3_seasonality": [_SENTINEL],
            "seq3_open3_seasonality": [_SENTINEL],
        },
        index=pd.DatetimeIndex([snapshot_date], name="date"),
    )
    feature_store.write("AAPL", frame)

    row = prepare_one(
        "AAPL",
        quotes={"AAPL": _quote(open_price=102.0, prev_close=100.0)},
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=as_of,
        spy_open=None,
        spy_prev_close=None,
        blocked=set(),
        excluded_features=frozenset({"seq2_open3_seasonality"}),
    )

    assert row.row is not None
    assert row.row.iloc[0]["seq2_open3_seasonality"] == pytest.approx(_SENTINEL)
    assert row.row.iloc[0]["seq3_open3_seasonality"] != pytest.approx(_SENTINEL)


def test_prepare_live_rows_reads_blacklist_once_and_merges_thread_buckets(tmp_path, monkeypatch):
    from stock_picker.training import live_rows as live_rows_mod

    feature_store, price_store = _stores(tmp_path)
    calls = {"n": 0}

    def _blocked():
        calls["n"] += 1
        return {"TDTH"}

    monkeypatch.setattr(live_rows_mod, "blacklisted_tickers", _blocked)

    quotes = {}
    for ticker in ["AAA", "BBB", "CCC", "DDD"]:
        _seed_features(feature_store, ticker)
        quotes[ticker] = _quote()
    quotes["TDTH"] = _quote()

    rows = prepare_live_rows(
        ["AAA", "BBB", "CCC", "DDD", "TDTH"],
        quotes=quotes,
        earnings=set(),
        feature_store=feature_store,
        price_store=price_store,
        as_of=_AS_OF,
        spy_open=None,
        spy_prev_close=None,
        bucket_size=2,
        workers=3,
    )

    assert calls["n"] == 1
    by_ticker = {row.ticker: row for row in rows}
    assert set(by_ticker) == {"AAA", "BBB", "CCC", "DDD", "TDTH"}
    assert by_ticker["TDTH"].skipped["reason"] == "blacklisted"
    for ticker in ["AAA", "BBB", "CCC", "DDD"]:
        assert by_ticker[ticker].row is not None
        assert by_ticker[ticker].snapshot_date == _SNAPSHOT.isoformat()
