"""Scenario serving must expose assumptions and reject hindsight/stale basis."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from stock_picker.ingestion.massive_overnight import VerifiedCurrentOpen, VerifiedOvernightBars
from stock_picker.training import overnight_service
from stock_picker.training.overnight_model import (
    DayModelOutputs, MODEL_FEATURE_COLUMNS, MODEL_FEATURE_VERSION, OvernightModel,
)


class AssumptionSensitiveBooster:
    def predict(self, frame):
        return np.array([0.01 + 0.5 * value for value in frame["assumed_day_return"]])


def inputs():
    session = date(2026, 1, 14)
    dates = pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        "2026-01-09", "2026-01-12", "2026-01-13",
    ])
    history = pd.DataFrame({"Close": [10.0] * len(dates)}, index=dates)
    provenance = pd.DataFrame({
        "source": "massive", "basis": "raw", "action_source": "massive_actions",
        "corporate_action": "verified_none",
    }, index=dates)
    model = OvernightModel(
        AssumptionSensitiveBooster(), overnight_service.RAW_CONTRACT.artifact_metadata(),
        MODEL_FEATURE_COLUMNS, MODEL_FEATURE_VERSION, date(2026, 1, 12),
        date(2026, 1, 13), (), 0.02, 0.2,
        day_fit_model=object(), day_rank_model=object(),
        day_model_trained_through=date(2026, 1, 13),
    )
    observed = VerifiedCurrentOpen(
        9.0, 10.0, 9.5, datetime(2026, 1, 14, tzinfo=timezone.utc),
        datetime(2026, 1, 14, tzinfo=timezone.utc), "verified_none",
    )
    return session, VerifiedOvernightBars(history, provenance, (), ()), observed, model


def test_three_close_scenarios_recompute_features_and_report_costs(monkeypatch):
    monkeypatch.setattr(
        overnight_service, "score_day_models", lambda *_: DayModelOutputs(0.02, 0.4, 0.01, 0.3),
    )
    session, verified, observed, model = inputs()
    result = overnight_service.build_forecast_cases(
        ticker="AAA", session=session, verified=verified, current=observed,
        day_row=pd.DataFrame({"sample": [1.0]}), model=model,
        assumed_close=9.5, step=0.05, shares=100,
        exit_today_cost_per_share=0.01, exit_next_open_cost_per_share=0.03,
    )
    assert [case.label for case in result.cases] == ["primary", "lower", "higher"]
    assert [case.forecast.assumed_close for case in result.cases] == pytest.approx([9.5, 9.45, 9.55])
    assert result.cases[2].forecast.predicted_gap > result.cases[0].forecast.predicted_gap
    assert result.cases[0].after_cost_difference_per_share == pytest.approx(
        result.cases[0].forecast.difference_per_share - 0.02
    )
    assert result.cases[0].features["assumed_close"] == 9.5


def test_missing_pinned_models_and_price_basis_mismatch_fail_closed(monkeypatch):
    monkeypatch.setattr(
        overnight_service, "score_day_models", lambda *_: DayModelOutputs(0.02, 0.4, 0.01, 0.3),
    )
    session, verified, observed, model = inputs()
    common = dict(
        ticker="AAA", session=session, verified=verified, current=observed,
        day_row=pd.DataFrame({"sample": [1.0]}), model=model, assumed_close=9.5,
    )
    model.day_fit_model = None
    with pytest.raises(ValueError, match="pin"):
        overnight_service.build_forecast_cases(**common)
    model.day_fit_model = object()
    wrong = VerifiedCurrentOpen(9.0, 10.05, None, None, observed.fetched_at, "verified_none")
    with pytest.raises(ValueError, match="previous close"):
        overnight_service.build_forecast_cases(**(common | {"current": wrong}))
    model.day_model_trained_through = session
    with pytest.raises(ValueError, match="saw the scenario"):
        overnight_service.build_forecast_cases(**common)
