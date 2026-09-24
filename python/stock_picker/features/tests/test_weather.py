import pandas as pd

from stock_picker.features.weather import build_weather_features, yesterday_weather_row


def test_yesterday_weather_is_calendar_prior_day():
    nyc = pd.DataFrame(
        {"tmax_c": [10.0, 28.0], "tmin_c": [1.0, 12.0], "precip_mm": [0.0, 12.0],

         "snow_cm": [0.0, 0.0], "hdd": [8.0, 0.0], "cdd": [0.0, 4.0]},
        index=pd.to_datetime(["2026-09-21", "2026-09-22"]),
    )
    row = yesterday_weather_row(nyc, "2026-09-23")
    assert row["weather_nyc_tmax_yday"] == 28.0
    assert row["weather_nyc_precip_yday"] == 12.0


def test_build_weather_features_aligns_each_session_to_calendar_yesterday():
    nyc = pd.DataFrame(
        {"tmax_c": [10.0, 28.0], "tmin_c": [1.0, 12.0], "precip_mm": [0.0, 0.0],
         "snow_cm": [0.0, 0.0], "hdd": [8.0, 0.0], "cdd": [0.0, 4.0]},
        index=pd.to_datetime(["2026-09-21", "2026-09-22"]),
    )
    index = pd.to_datetime(["2026-09-22", "2026-09-23"])
    frame = build_weather_features(index, nyc)
    assert frame.loc[index[0], "weather_nyc_tmax_yday"] == 10.0
    assert frame.loc[index[1], "weather_nyc_tmax_yday"] == 28.0
