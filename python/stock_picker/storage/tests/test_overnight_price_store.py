"""Overnight storage keeps bars and certified evidence together."""

import pandas as pd
import pytest

from stock_picker.storage.overnight_price_store import OvernightPriceStore


class FakeXNYS:
    name = "XNYS"

    def is_session(self, session):
        return session in {"2026-10-01", "2026-10-02", "2026-10-05"}


def sample():
    index = pd.DatetimeIndex(["2026-10-01", "2026-10-02"], name="Date")
    bars = pd.DataFrame(
        {"Open": [10.0, 10.1], "High": [11.0, 11.0], "Low": [9.0, 9.1],
         "Close": [10.5, 10.6], "Volume": [1000.0, 1100.0]}, index=index,
    )
    provenance = pd.DataFrame(
        {"source": "massive", "basis": "raw", "action_source": "massive_actions",
         "corporate_action": "verified_none", "action_event_ids": "",
         "action_verified_from": "2026-10-01", "action_verified_to": "2026-10-02",
         "actions_fetched_at": "2026-10-06T18:00:00+00:00",
         "bars_fetched_at": "2026-10-06T18:00:00+00:00"}, index=index,
    )
    return bars, provenance


def test_round_trip_keeps_same_index_and_evidence(tmp_path):
    bars, provenance = sample()
    store = OvernightPriceStore(tmp_path, calendar=FakeXNYS())
    store.write("AAPL", bars, provenance)
    stored_bars, stored_provenance = store.read("AAPL")
    pd.testing.assert_frame_equal(stored_bars, bars, check_freq=False)
    pd.testing.assert_frame_equal(stored_provenance, provenance, check_freq=False)
    assert list(tmp_path.iterdir()) == [tmp_path / "AAPL.parquet"]


def test_rejects_unverified_or_misaligned_evidence(tmp_path):
    bars, provenance = sample()
    store = OvernightPriceStore(tmp_path, calendar=FakeXNYS())
    provenance.loc["2026-10-02", "action_verified_to"] = "2026-10-01"
    with pytest.raises(ValueError, match="does not cover"):
        store.write("AAPL", bars, provenance)
    assert not list(tmp_path.iterdir())

    _, provenance = sample()
    provenance.loc["2026-10-02", "basis"] = "adjusted"
    with pytest.raises(ValueError, match="raw-basis"):
        store.write("AAPL", bars, provenance)

    _, provenance = sample()
    with pytest.raises(ValueError, match="share ordered"):
        store.write("AAPL", bars, provenance.iloc[:1])


def test_rejects_ticker_path_traversal(tmp_path):
    bars, provenance = sample()
    with pytest.raises(ValueError, match="ticker"):
        OvernightPriceStore(tmp_path, calendar=FakeXNYS()).write("../AAPL", bars, provenance)


@pytest.mark.parametrize(("column", "value", "message"), [
    ("Close", float("nan"), "finite numbers"),
    ("High", 9.9, "inconsistent"),
    ("Volume", -1, "inconsistent"),
    ("Open", "bad", "finite numbers"),
    ("Open", 10**1000, "finite numbers"),
])
def test_rejects_invalid_ohlcv(tmp_path, column, value, message):
    bars, provenance = sample()
    if isinstance(value, str) or isinstance(value, int) and value > 10**100:
        bars[column] = bars[column].astype(object)
    bars.loc["2026-10-01", column] = value
    with pytest.raises(ValueError, match=message):
        OvernightPriceStore(tmp_path, calendar=FakeXNYS()).write("AAPL", bars, provenance)


def test_rejects_non_session_index(tmp_path):
    bars, provenance = sample()
    bars.index = pd.DatetimeIndex(["2026-10-01", "2026-10-03"], name="Date")
    provenance.index = bars.index
    with pytest.raises(ValueError, match="XNYS session"):
        OvernightPriceStore(tmp_path, calendar=FakeXNYS()).write("AAPL", bars, provenance)


def test_default_xnys_calendar_rejects_labor_day(tmp_path):
    pytest.importorskip("exchange_calendars")
    bars, provenance = sample()
    bars.index = pd.DatetimeIndex(["2026-09-07", "2026-10-01"], name="Date")
    provenance.index = bars.index
    with pytest.raises(ValueError, match="XNYS session"):
        OvernightPriceStore(tmp_path).write("AAPL", bars, provenance)


@pytest.mark.parametrize(("status", "ids"), [
    ("verified_none", "split:event-1"),
    ("split", ""),
    ("dividend", "split:event-1"),
    ("split", "nonsense"),
])
def test_rejects_incoherent_action_evidence(tmp_path, status, ids):
    bars, provenance = sample()
    provenance.loc["2026-10-02", "corporate_action"] = status
    provenance.loc["2026-10-02", "action_event_ids"] = ids
    with pytest.raises(ValueError, match="action"):
        OvernightPriceStore(tmp_path, calendar=FakeXNYS()).write("AAPL", bars, provenance)


def test_rejects_naive_fetch_timestamp(tmp_path):
    bars, provenance = sample()
    provenance.loc["2026-10-01", "actions_fetched_at"] = "2026-10-06T18:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        OvernightPriceStore(tmp_path, calendar=FakeXNYS()).write("AAPL", bars, provenance)


def test_rejects_noncanonical_verification_window(tmp_path):
    bars, provenance = sample()
    provenance.loc["2026-10-01", "action_verified_from"] = "20261001"
    with pytest.raises(ValueError, match="verification window"):
        OvernightPriceStore(tmp_path, calendar=FakeXNYS()).write("AAPL", bars, provenance)


def test_existing_longer_history_cannot_be_silently_replaced(tmp_path):
    bars, provenance = sample()
    store = OvernightPriceStore(tmp_path, calendar=FakeXNYS())
    store.write("AAPL", bars, provenance)
    with pytest.raises(FileExistsError, match="explicit overwrite"):
        store.write("AAPL", bars.iloc[:1], provenance.iloc[:1])
    with pytest.raises(ValueError, match="discard existing"):
        store.write("AAPL", bars.iloc[:1], provenance.iloc[:1], overwrite=True)
    restored, _ = store.read("AAPL")
    assert len(restored) == 2
    store.write("AAPL", bars, provenance, overwrite=True)


def test_split_and_dividend_ids_can_coexist_on_one_session(tmp_path):
    bars, provenance = sample()
    provenance.loc["2026-10-02", "corporate_action"] = "split"
    provenance.loc["2026-10-02", "action_event_ids"] = "split:s1|dividend:d1"
    store = OvernightPriceStore(tmp_path, calendar=FakeXNYS())
    store.write("AAPL", bars, provenance)
    _, read_back = store.read("AAPL")
    assert read_back.loc["2026-10-02", "action_event_ids"] == "split:s1|dividend:d1"
