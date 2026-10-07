"""The next-open model consumes morning scores only with as-of provenance."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.storage.model_store import ModelStore
from stock_picker.training.overnight_dataset import (
    CurrentOpenProvenance, FEATURE_COLUMNS, LABEL_COLUMN, PriceContract, ScenarioBuild,
)
from stock_picker.training.overnight_day_scores import generate_historical_day_scores
from stock_picker.training.overnight_job import train_and_persist_overnight
from stock_picker.training.overnight_model import (
    DAY_OUTPUT_COLUMNS,
    FIT_RESIDUAL_COLUMN,
    MODEL_FEATURE_VERSION,
    MODEL_FEATURE_COLUMNS,
    DayModelOutputs,
    OvernightModel,
    attach_historical_day_scores,
    add_model_derived_features,
    forecast_assumed_close,
    forecast_if_closes_at,
    score_day_models,
    train_overnight_model,
)


CONTRACT = PriceContract("massive", "raw", "massive_actions")


def make_rows() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=26)
    rows = []
    for day_index, session in enumerate(dates):
        for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
            prior_close = 20 + ticker_index * 10 + day_index * 0.1
            today_open = prior_close * (1 + (day_index % 5 - 2) * 0.002)
            assumed_close = today_open * (1 + (day_index % 7 - 3) * 0.003)
            rows.append({
                "ticker": ticker,
                "date": session,
                "prior_close": prior_close,
                "prior_return_1d": 0.001 * (day_index % 3 - 1),
                "prior_return_5d": 0.002 * (day_index % 4 - 2),
                "prior_volatility_5d": 0.02,
                "today_open": today_open,
                "today_open_gap": today_open / prior_close - 1,
                "assumed_close": assumed_close,
                "assumed_day_return": assumed_close / today_open - 1,
                "assumed_vs_prior_close": assumed_close / prior_close - 1,
                "weekday": float(session.weekday()),
                DAY_OUTPUT_COLUMNS[0]: 0.001 * (ticker_index - 1),
                DAY_OUTPUT_COLUMNS[1]: 0.2 + ticker_index * 0.1,
                DAY_OUTPUT_COLUMNS[2]: 0.002 * (day_index % 4 - 2),
                DAY_OUTPUT_COLUMNS[3]: 0.3 * (ticker_index - 1),
                LABEL_COLUMN: 0.001 * (day_index % 5 - 2) + ticker_index * 0.0002,
            })
    return pd.DataFrame(rows)


def test_contract_has_fifteen_features_and_fit_prediction_is_a_real_column():
    assert MODEL_FEATURE_COLUMNS[:10] == FEATURE_COLUMNS
    assert len(MODEL_FEATURE_COLUMNS) == 15
    assert "day_fit_predicted_return" in MODEL_FEATURE_COLUMNS
    assert "assumed_close" in MODEL_FEATURE_COLUMNS
    assert FIT_RESIDUAL_COLUMN in MODEL_FEATURE_COLUMNS


def test_fit_residual_uses_assumed_price_and_cannot_be_spoofed():
    rows = make_rows().iloc[:1]
    enriched = add_model_derived_features(rows)
    assert enriched.iloc[0][FIT_RESIDUAL_COLUMN] == pytest.approx(
        rows.iloc[0][DAY_OUTPUT_COLUMNS[0]] - rows.iloc[0]["assumed_day_return"]
    )
    with pytest.raises(ValueError, match="does not match"):
        add_model_derived_features(enriched.assign(**{FIT_RESIDUAL_COLUMN: 9.0}))


def test_historical_scores_require_prior_training_and_full_coverage():
    rows = make_rows().iloc[:2].drop(columns=list(DAY_OUTPUT_COLUMNS))
    scores = make_rows().iloc[:2][["ticker", "date", *DAY_OUTPUT_COLUMNS]].copy()
    scores["trained_through"] = pd.Timestamp("2026-01-02")
    joined = attach_historical_day_scores(rows, scores)
    assert joined[DAY_OUTPUT_COLUMNS[0]].tolist() == [-0.001, 0.0]

    scores.loc[0, "trained_through"] = scores.loc[0, "date"]
    with pytest.raises(ValueError, match="strictly earlier"):
        attach_historical_day_scores(rows, scores)
    intraday_rows, intraday_scores = rows.copy(), scores.copy()
    intraday_rows.loc[0, "date"] = pd.Timestamp("2026-01-05 12:00")
    intraday_scores.loc[0, "date"] = pd.Timestamp("2026-01-05 12:00")
    intraday_scores.loc[0, "trained_through"] = pd.Timestamp("2026-01-05 09:00")
    with pytest.raises(ValueError, match="strictly earlier"):
        attach_historical_day_scores(intraday_rows, intraday_scores)
    with pytest.raises(ValueError, match="unique"):
        attach_historical_day_scores(rows, pd.concat([scores, scores.iloc[[0]]]))
    with pytest.raises(ValueError, match="every overnight row"):
        attach_historical_day_scores(rows, scores.iloc[:1])


def test_train_evaluate_persist_and_forecast_scenario(tmp_path):
    rows = make_rows()
    model = train_overnight_model(rows, CONTRACT, n_splits=2, rounds=5, params={"min_data_in_leaf": 2})
    assert model.feature_columns == MODEL_FEATURE_COLUMNS
    assert model.contract["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert len(model.folds) == 2
    assert all(fold.train_through < fold.test_start for fold in model.folds)
    assert all(fold.n_rows > 0 and fold.zero_gap_mae >= 0 for fold in model.folds)
    assert all(fold.ticker_mean_gap_mae >= 0 for fold in model.folds)
    assert model.open_abs_error_p90 >= 0

    store = ModelStore(data_dir=tmp_path)
    store.write("next_open_from_assumed_close", model)
    restored = store.read("next_open_from_assumed_close")
    last = rows.iloc[-1]
    outputs = DayModelOutputs(*(float(last[column]) for column in DAY_OUTPUT_COLUMNS))
    scenario = ScenarioBuild(last[list(FEATURE_COLUMNS)], date(2026, 2, 11), None)
    result = forecast_assumed_close(
        scenario, outputs, restored, session=date(2026, 2, 10), contract=CONTRACT,
    )
    assert result.projected_open == pytest.approx(result.assumed_close * (1 + result.predicted_gap))
    assert result.difference_per_share == pytest.approx(result.projected_open - result.assumed_close)
    with pytest.raises(ValueError, match="seen the scenario"):
        forecast_assumed_close(scenario, outputs, restored, session=model.trained_through, contract=CONTRACT)


def test_same_scoring_path_is_used_for_saved_fit_rank_and_svm(monkeypatch):
    from stock_picker.training import overnight_model

    fit = SimpleNamespace(stacked_svm_estimators={"svr_oof_pred": object(), "svc_direction_margin": object()})
    rank = SimpleNamespace()
    row = pd.DataFrame({"signal": [1.0]})
    monkeypatch.setattr(
        overnight_model, "score_stacked_svm",
        lambda frame, estimators, outputs: pd.DataFrame({"svr_oof_pred": [0.02], "svc_direction_margin": [0.7]}),
    )
    monkeypatch.setattr(
        overnight_model, "predict_ensemble",
        lambda ensemble, frame: np.array([0.03 if ensemble is fit else 0.4]),
    )
    values = score_day_models(row, fit, rank)
    assert values.as_features() == dict(zip(DAY_OUTPUT_COLUMNS, (0.03, 0.4, 0.02, 0.7)))


def test_full_scenario_recomputes_close_derived_inputs(monkeypatch):
    from stock_picker.training import overnight_model

    class AssumptionSensitiveBooster:
        def predict(self, frame):
            return 0.01 + 0.5 * frame["assumed_day_return"].to_numpy()

    sessions = pd.to_datetime([
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        "2026-01-09", "2026-01-12", "2026-01-13",
    ])
    prior = pd.DataFrame({"Close": [10.0] * len(sessions)}, index=sessions)
    provenance = pd.DataFrame({
        "source": "massive", "basis": "raw", "action_source": "massive_actions",
        "corporate_action": "verified_none",
    }, index=sessions)
    artifact = OvernightModel(
        booster=AssumptionSensitiveBooster(), contract=CONTRACT.artifact_metadata(),
        feature_columns=MODEL_FEATURE_COLUMNS, feature_version=MODEL_FEATURE_VERSION,
        trained_through=date(2026, 1, 12), label_observed_on=date(2026, 1, 13), folds=(),
        gap_abs_error_p90=0.02, open_abs_error_p90=0.2,
    )
    monkeypatch.setattr(overnight_model, "score_day_models", lambda *args: DayModelOutputs(0.02, 0.4, 0.01, 0.3))
    common = dict(
        prior_history=prior, provenance=provenance, session=date(2026, 1, 14),
        today_open=9.0,
        current_open_provenance=CurrentOpenProvenance("massive", "raw", "massive_actions", "verified_none"),
        open_known_day_row=pd.DataFrame({"signal": [1.0]}),
        fit_model=SimpleNamespace(), rank_model=SimpleNamespace(),
        fit_trained_through=date(2026, 1, 13), rank_trained_through=date(2026, 1, 13),
        overnight_model=artifact, contract=CONTRACT,
    )
    low = forecast_if_closes_at(**common, assumed_close=9.50)
    high = forecast_if_closes_at(**common, assumed_close=9.55)
    assert high.predicted_gap > low.predicted_gap
    assert low.next_session == date(2026, 1, 15)
    with pytest.raises(ValueError, match="fitted before"):
        forecast_if_closes_at(**(common | {"fit_trained_through": date(2026, 1, 14)}), assumed_close=9.50)


def test_historical_generator_tags_each_score_with_earlier_fold_cutoff(monkeypatch, tmp_path):
    from stock_picker.training import overnight_day_scores

    dates = pd.bdate_range("2026-01-05", periods=12)
    day_rows = pd.DataFrame({
        "ticker": ["AAA"] * len(dates),
        "date": dates,
        "label_day_session_return": [0.01] * len(dates),
    })
    fit_model = SimpleNamespace(stacked_svm_estimators={"svr_oof_pred": object(), "svc_direction_margin": object()})
    monkeypatch.setattr(
        overnight_day_scores, "run_walk_forward",
        lambda frame, **kwargs: [SimpleNamespace(model=fit_model)] * 2,
    )
    monkeypatch.setattr(overnight_day_scores, "train_ensemble", lambda frame, specs: object())
    monkeypatch.setattr(
        overnight_day_scores, "score_day_model_frame",
        lambda frame, fit, rank: pd.DataFrame(
            {name: np.ones(len(frame)) for name in DAY_OUTPUT_COLUMNS}, index=frame.index,
        ),
    )
    scores = generate_historical_day_scores(
        day_rows,
        fit_specs=[SimpleNamespace(model_type="lightgbm")],
        rank_spec=SimpleNamespace(model_type="lightgbm_rank"),
        tracking_dir=tmp_path,
        n_splits=2,
    )
    assert len(scores) == 8
    assert (scores["trained_through"] < scores["date"]).all()
    assert set(DAY_OUTPUT_COLUMNS).issubset(scores.columns)


def test_job_publishes_only_after_prior_trained_scores_and_model_fit(monkeypatch, tmp_path):
    from stock_picker.training import overnight_job

    rows = make_rows().drop(columns=list(DAY_OUTPUT_COLUMNS))
    scores = make_rows()[["ticker", "date", *DAY_OUTPUT_COLUMNS]].copy()
    scores["trained_through"] = scores["date"] - pd.Timedelta(days=1)
    first_scored = scores["date"].sort_values().unique()[4]
    scores = scores.loc[scores["date"] >= first_scored]
    monkeypatch.setattr(overnight_job, "build_verified_overnight_rows", lambda verified, contract: (rows, {"target_not_observed": 3}))
    monkeypatch.setattr(overnight_job, "generate_historical_day_scores", lambda *args, **kwargs: scores)
    monkeypatch.setattr(overnight_job, "train_overnight_model", lambda frame, contract, **kwargs: SimpleNamespace(n_rows=len(frame)))
    store = ModelStore(data_dir=tmp_path / "models")
    result = train_and_persist_overnight(
        {}, pd.DataFrame(), fit_specs=[], rank_spec=SimpleNamespace(model_type="lightgbm_rank"),
        contract=CONTRACT, model_store=store, tracking_dir=tmp_path, n_splits=2,
    )
    assert result.label_rows == len(rows)
    assert result.warmup_rows == 12
    assert result.scored_rows == len(rows) - 12
    assert store.exists("next_open_from_assumed_close")

    scores.loc[scores.index[0], "trained_through"] = scores.loc[scores.index[0], "date"]
    with pytest.raises(ValueError, match="strictly earlier"):
        train_and_persist_overnight(
            {}, pd.DataFrame(), fit_specs=[], rank_spec=SimpleNamespace(model_type="lightgbm_rank"),
            contract=CONTRACT, model_store=store, tracking_dir=tmp_path, n_splits=2,
        )
