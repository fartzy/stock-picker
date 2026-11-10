from datetime import datetime, timedelta
import math

import numpy as np
import pytest

from stock_picker.features.intraday_entry import (
    FEATURE_COLUMNS, MINUTE_MS, build_intraday_entry_features, normalize_rank_score,
)


def _state(*, day="2026-11-10", opened="14:30", closed="21:00", minutes=30):
    start = datetime.fromisoformat(f"{day}T{opened}:00+00:00")
    end = datetime.fromisoformat(f"{day}T{closed}:00+00:00")
    bars = [dict(t=round(start.timestamp() * 1000) + i * MINUTE_MS,
                 o=100 + i / 10, h=101 + i / 10, l=99 + i / 10,
                 c=100.5 + i / 10, v=100 + i, vw=100.25 + i / 10)
            for i in range(40)]
    args = dict(observed_at=start + timedelta(minutes=minutes), session_open_at=start,
                session_close_at=end, entry_price=96., session_open=100., previous_close=101.,
                rank=1, rank_score_z=1.5, morning_features={"volatility_32d": .02})
    return bars, args


def _assert_same(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert (math.isnan(value) and math.isnan(right[key])) or value == right[key], key


def test_future_and_incomplete_minute_ohlcv_cannot_change_features():
    bars, args = _state()
    expected = build_intraday_entry_features(bars, **args)
    changed = [dict(b) for b in bars]
    for bar in changed[30:]:
        bar.update(o=-1, h=1e12, l=-1e12, c=1e9, v=1e15, vw=1e9)
    # The current minute is still incomplete at 10:00:45 ET.
    args["observed_at"] += timedelta(seconds=45)
    later = build_intraday_entry_features(changed, **args)
    original_later = build_intraday_entry_features(bars, **args)
    _assert_same(later, original_later)
    args["observed_at"] -= timedelta(seconds=45)
    _assert_same(build_intraday_entry_features(changed, **args), expected)
    assert expected["return_from_open"] == pytest.approx(-.04)
    assert expected["minutes_since_low"] == 0


def test_premarket_is_excluded_and_numpy_scalars_are_numeric():
    bars, args = _state()
    expected = build_intraday_entry_features(bars, **args)
    bars.insert(0, dict(t=bars[0]["t"] - MINUTE_MS, o=-1, h=-1, l=-1, c=-1))
    args.update(entry_price=np.float64(96), rank=np.int64(1),
                morning_features={"volatility_32d": np.float32(.02)})
    actual = build_intraday_entry_features(bars, **args)
    assert actual["morning_rank"] == 1
    assert actual["prior_volatility_32d"] == pytest.approx(.02)
    assert actual["return_from_open_in_prior_vol"] == pytest.approx(-2)
    assert actual["range_so_far"] == expected["range_so_far"]


def test_gap_is_not_forward_filled_and_latest_completed_minute_is_required():
    bars, args = _state()
    missing = build_intraday_entry_features([b for i, b in enumerate(bars) if i != 27], **args)
    assert missing["observed_bar_fraction"] == pytest.approx(29 / 30)
    assert math.isnan(missing["return_5m"])
    assert math.isnan(missing["realized_volatility_15m"])
    assert not math.isnan(missing["return_1m"])
    with pytest.raises(ValueError, match="missing_last_completed_minute"):
        build_intraday_entry_features([b for i, b in enumerate(bars) if i != 29], **args)


@pytest.mark.parametrize(("day", "opened", "closed", "remaining"), [
    ("2026-07-02", "13:30", "20:00", 360),
    ("2026-11-10", "14:30", "21:00", 360),
    ("2026-11-27", "14:30", "18:00", 180),
])
def test_timezone_and_early_close_use_actual_session_bounds(day, opened, closed, remaining):
    bars, args = _state(day=day, opened=opened, closed=closed)
    row = build_intraday_entry_features(bars, **args)
    assert row["minutes_since_open"] == 30
    assert row["minutes_to_close"] == remaining
    assert row["elapsed_session_fraction"] == pytest.approx(30 / (30 + remaining))


def test_unknown_previous_close_and_volume_are_not_fabricated():
    bars, args = _state()
    bars[1].pop("v")
    args["previous_close"] = None
    row = build_intraday_entry_features(bars, **args)
    assert row["previous_close_known"] == 0
    for field in ("opening_gap", "return_from_previous_close", "log_observed_volume", "entry_vs_observed_vwap"):
        assert math.isnan(row[field])


def test_rank_raw_scale_does_not_transfer_to_features():
    bars, args = _state()
    _assert_same(build_intraday_entry_features(bars, rank_score=.5, **args),
                 build_intraday_entry_features(bars, rank_score=5000., **args))
    assert normalize_rank_score(5., [5., 4., 3.]) == pytest.approx(normalize_rank_score(500., [500., 400., 300.]))
    assert math.isnan(normalize_rank_score(1., [1., 1.]))
    assert "rank_score" not in FEATURE_COLUMNS


def test_invalid_or_duplicate_observed_bars_and_naive_clock_are_rejected():
    bars, args = _state()
    with pytest.raises(ValueError, match="duplicate minute"):
        build_intraday_entry_features([*bars, bars[1]], **args)
    bars[1]["l"] = 900
    with pytest.raises(ValueError, match="invalid completed-minute"):
        build_intraday_entry_features(bars, **args)
    args["observed_at"] = args["observed_at"].replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        build_intraday_entry_features([], **args)
