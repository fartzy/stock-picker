import pandas as pd
import pytest

from stock_picker.features import breadth


def _history(index: pd.DatetimeIndex, close: list[float], volume: float) -> pd.DataFrame:
    prices = pd.Series(close, index=index)
    return pd.DataFrame({
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Close": prices,
        "Volume": volume,
    }, index=index)


def test_breadth_respects_membership_and_does_not_use_future_closes(monkeypatch):
    monkeypatch.setattr(breadth, "MIN_BREADTH_NAMES", 2)
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    up = _history(dates, [100.0 + day for day in range(25)], 10.0)
    down = _history(dates, [125.0 - day for day in range(25)], 30.0)
    histories = {"UP": up, "DOWN": down}
    first_seen = {"UP": dates[0].date().isoformat(), "DOWN": dates[22].date().isoformat()}

    result = breadth.build_breadth_features(histories, first_seen)

    assert result.loc[dates[21]].isna().all()
    assert result.loc[dates[22], "market_advance_fraction"] == pytest.approx(0.5)
    assert result.loc[dates[22], "market_above_sma_20d_fraction"] == pytest.approx(0.5)
    assert result.loc[dates[22], "market_up_volume_fraction"] == pytest.approx(0.25)
    assert result.loc[dates[22], "market_return_dispersion_1d"] > 0

    changed = {**histories, "DOWN": down.copy()}
    changed["DOWN"].loc[dates[-1], "Close"] = 999.0
    revised = breadth.build_breadth_features(changed, first_seen)
    pd.testing.assert_frame_equal(result.iloc[:-1], revised.iloc[:-1])


def test_breadth_requires_a_recorded_first_seen():
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    result = breadth.build_breadth_features(
        {"UP": _history(dates, [100.0 + day for day in range(25)], 10.0)},
        {},
    )

    assert list(result.columns) == list(breadth.BREADTH_COLUMNS)
    assert result.empty
