"""Scenario serving must expose assumptions and reject hindsight/stale basis."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.ingestion.massive_overnight import VerifiedCurrentOpen, VerifiedOvernightBars
from stock_picker.training import overnight_service
from stock_picker.training.overnight_model import (
    DayModelOutputs, MODEL_FEATURE_COLUMNS, MODEL_FEATURE_VERSION, OvernightModel,
    STACKED_FEATURE_COLUMNS, STACKED_FEATURE_VERSION,
)
from stock_picker.training.overnight_variants import VARIANT_OUTPUT_COLUMNS


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


def test_old_artifact_stays_ready_but_stacked_artifact_requires_all_pinned_variants():
    _, _, _, model = inputs()
    # A prior pickle has no variant field at all; only the 15 inputs apply.
    del model.day_variant_models
    assert overnight_service.model_summary(model)["serving_inputs_pinned"] is True
    model.feature_columns = STACKED_FEATURE_COLUMNS
    model.feature_version = STACKED_FEATURE_VERSION
    assert overnight_service.model_summary(model)["serving_inputs_pinned"] is False
    model.day_variant_models = {column: object() for column in VARIANT_OUTPUT_COLUMNS[:-1]}
    assert overnight_service.model_summary(model)["serving_inputs_pinned"] is False
    model.day_variant_models[VARIANT_OUTPUT_COLUMNS[-1]] = object()
    model.day_variant_trained_through = date(2026, 1, 13)
    assert overnight_service.model_summary(model)["serving_inputs_pinned"] is True


def test_stacked_inputs_require_immediately_previous_exchange_snapshot(monkeypatch):
    # Tuesday follows the Labor Day closure, so Friday—not Monday—is prior.
    session = date(2026, 9, 8)
    prepared = SimpleNamespace(
        row=pd.DataFrame({"signal": [1.0]}), snapshot_date="2026-09-04",
    )
    monkeypatch.setattr(overnight_service, "prepare_one", lambda *args, **kwargs: prepared)

    class Provider:
        def fetch(self, *args):
            return object()

        def fetch_current_open(self, *args):
            return SimpleNamespace(open=10.0, previous_close=9.5, last_trade=10.0)

    provider = Provider()
    _, _, row = overnight_service.fetch_forecast_inputs(
        "AAA", session, client=provider, feature_store=object(), price_store=object(),
        require_prior_session_snapshot=True,
    )
    assert row.equals(prepared.row)

    prepared.snapshot_date = "2026-09-03"
    with pytest.raises(ValueError, match="prior XNYS feature snapshot 2026-09-04"):
        overnight_service.fetch_forecast_inputs(
            "AAA", session, client=provider, feature_store=object(), price_store=object(),
            require_prior_session_snapshot=True,
        )

    # Serving a saved 15-input artifact must retain its existing freshness rule.
    _, _, row = overnight_service.fetch_forecast_inputs(
        "AAA", session, client=provider, feature_store=object(), price_store=object(),
    )
    assert row.equals(prepared.row)


@pytest.mark.parametrize("stacked", [False, True])
def test_service_enforces_prior_snapshot_only_for_stacked_artifact(monkeypatch, stacked):
    session, _, _, model = inputs()
    if stacked:
        model.feature_columns = STACKED_FEATURE_COLUMNS
        model.feature_version = STACKED_FEATURE_VERSION
        model.day_variant_models = {column: object() for column in VARIANT_OUTPUT_COLUMNS}
        model.day_variant_trained_through = date(2026, 1, 13)

    class Store:
        def exists(self, name):
            return True

        def read(self, name):
            return model

    flags = []

    def fetch(*args, **kwargs):
        flags.append(kwargs["require_prior_session_snapshot"])
        return object(), object(), pd.DataFrame({"signal": [1.0]})

    monkeypatch.setattr(overnight_service, "fetch_forecast_inputs", fetch)
    monkeypatch.setattr(overnight_service, "build_forecast_cases", lambda **kwargs: "served")
    assert overnight_service.serve_overnight_forecast(
        ticker="AAA", session=session, assumed_close=9.5, model_store=Store(),
    ) == "served"
    assert flags == [stacked]
