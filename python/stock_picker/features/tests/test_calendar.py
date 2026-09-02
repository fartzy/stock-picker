import pandas as pd

from stock_picker.features.calendar import build_calendar_features


def test_build_calendar_features_matches_known_dates():
    history = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-31"]),
    )

    features = build_calendar_features(history)

    # 2026-03-02 is a Monday (day_of_week=0), 2026-03-03 a Tuesday (1)
    assert features["day_of_week"].tolist() == [0, 1, 1]
    assert features["day_of_month"].tolist() == [2, 3, 31]
    assert features["month"].tolist() == [3, 3, 3]
