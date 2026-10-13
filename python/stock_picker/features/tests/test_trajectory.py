import pandas as pd

from stock_picker.features import trajectory


def _history(index: pd.DatetimeIndex, base_return: float) -> pd.DataFrame:
    close = [100.0]
    for day in range(1, len(index)):
        daily_return = base_return + 0.005 * (-1) ** day + 0.002 * (day % 3)
        close.append(close[-1] * (1 + daily_return))
    prices = pd.Series(close, index=index)
    return pd.DataFrame({
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Close": prices,
        "Volume": [100.0 + 10 * (day % 4) for day in range(len(index))],
    }, index=index)


def test_trajectory_scores_only_after_past_cluster_training(monkeypatch):
    monkeypatch.setattr(trajectory, "MIN_PATH_NAMES", 4)
    monkeypatch.setattr(trajectory, "CLUSTER_K", 2)
    dates = pd.date_range("2026-01-01", periods=36, freq="B")
    histories = {
        "A": _history(dates, 0.015),
        "B": _history(dates, 0.01),
        "C": _history(dates, -0.01),
        "D": _history(dates, -0.015),
    }
    first_seen = {name: dates[0].date().isoformat() for name in histories}

    result = trajectory.build_trajectory_features(histories, first_seen)

    assert result["A"].iloc[:24].isna().all().all()
    assert result["A"].iloc[-1]["path_cluster_distance"] >= 0
    assert 0 < result["A"].iloc[-1]["path_cluster_crowding"] <= 1

    changed = {**histories, "D": histories["D"].copy()}
    changed["D"].loc[dates[-1], "Close"] *= 1.5
    revised = trajectory.build_trajectory_features(changed, first_seen)
    for ticker in histories:
        pd.testing.assert_frame_equal(result[ticker].iloc[:-1], revised[ticker].iloc[:-1])


def test_trajectory_does_not_backdate_a_new_member(monkeypatch):
    monkeypatch.setattr(trajectory, "MIN_PATH_NAMES", 4)
    monkeypatch.setattr(trajectory, "CLUSTER_K", 2)
    dates = pd.date_range("2026-01-01", periods=36, freq="B")
    histories = {name: _history(dates, signal) for name, signal in {
        "A": 0.015, "B": 0.01, "C": -0.01, "D": -0.015,
    }.items()}
    first_seen = {name: dates[0].date().isoformat() for name in histories}
    first_seen["D"] = dates[-1].date().isoformat()

    result = trajectory.build_trajectory_features(histories, first_seen)

    assert result["A"].isna().all().all()
    assert result["D"].isna().all().all()
