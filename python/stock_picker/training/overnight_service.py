"""One close-conditioned forecast path shared by the API and later replay.

The serving adapter obtains verified Massive raw bars and a session-dated open.
The pure scenario function recomputes features for each hypothetical close.
It does not turn a point estimate into a hold/sell recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from math import isclose, isfinite

import exchange_calendars as xcals
import pandas as pd

from stock_picker.ingestion.massive_overnight import (
    MassiveOvernightClient,
    VerifiedCurrentOpen,
    VerifiedOvernightBars,
)
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.model_store import ModelStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.training.live_rows import prepare_one
from stock_picker.training.overnight_dataset import (
    CurrentOpenProvenance,
    PriceContract,
    build_scenario_features,
    next_expected_session,
)
from stock_picker.training.overnight_model import (
    DayModelOutputs,
    MODEL_NAME,
    OvernightForecast,
    OvernightModel,
    STACKED_FEATURE_COLUMNS,
    forecast_assumed_close,
    score_day_models,
)
from stock_picker.training.overnight_variants import VARIANT_OUTPUT_COLUMNS


RAW_CONTRACT = PriceContract("massive", "raw", "massive_actions")


@dataclass(frozen=True)
class ForecastCase:
    label: str
    forecast: OvernightForecast
    features: dict[str, float]
    after_cost_difference_per_share: float | None


@dataclass(frozen=True)
class OvernightScenarioResult:
    ticker: str
    session: date
    today_open: float
    last_trade: float | None
    last_trade_at: datetime | None
    quote_fetched_at: datetime
    step: float
    shares: float | None
    cases: tuple[ForecastCase, ...]
    day_outputs: DayModelOutputs
    model_trained_through: date
    model_label_observed_on: date
    model_feature_version: str
    evaluated_rows: int


def model_summary(model: OvernightModel | None) -> dict[str, object]:
    """Visible even without a selected artifact; never imply one exists."""
    from stock_picker.training.overnight_model import MODEL_FEATURE_COLUMNS, MODEL_FEATURE_VERSION

    if model is None:
        return {
            "available": False, "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "feature_version": MODEL_FEATURE_VERSION, "trained_through": None,
            "label_observed_on": None, "evaluated_rows": 0,
            "model_gap_mae": None, "unchanged_gap_mae": None,
            "ticker_mean_gap_mae": None, "day_model_source": None,
            "serving_inputs_pinned": False,
        }
    rows = sum(fold.n_rows for fold in model.folds)

    def average(name: str) -> float | None:
        return sum(getattr(fold, name) * fold.n_rows for fold in model.folds) / rows if rows else None

    pinned = all((
        getattr(model, "day_fit_model", None) is not None,
        getattr(model, "day_rank_model", None) is not None,
        getattr(model, "day_model_trained_through", None) is not None,
    ))
    if model.feature_columns == STACKED_FEATURE_COLUMNS:
        variants = getattr(model, "day_variant_models", None) or {}
        pinned = pinned and getattr(model, "day_variant_trained_through", None) is not None
        pinned = pinned and set(variants) == set(VARIANT_OUTPUT_COLUMNS) and all(
            variants[column] is not None for column in VARIANT_OUTPUT_COLUMNS
        )
    return {
        "available": True, "feature_columns": list(model.feature_columns),
        "feature_version": model.feature_version,
        "trained_through": model.trained_through.isoformat(),
        "label_observed_on": model.label_observed_on.isoformat(),
        "evaluated_rows": rows, "model_gap_mae": average("model_gap_mae"),
        "unchanged_gap_mae": average("zero_gap_mae"),
        "ticker_mean_gap_mae": average("ticker_mean_gap_mae"),
        "day_model_source": model.day_model_source,
        "serving_inputs_pinned": pinned,
    }


def build_forecast_cases(
    *, ticker: str, session: date, verified: VerifiedOvernightBars,
    current: VerifiedCurrentOpen, day_row: pd.DataFrame,
    model: OvernightModel, assumed_close: float, step: float | None = None,
    shares: float | None = None, exit_today_cost_per_share: float | None = None,
    exit_next_open_cost_per_share: float | None = None,
) -> OvernightScenarioResult:
    """Recompute all three scenarios against the same observed input snapshot."""
    if not isfinite(assumed_close) or assumed_close <= 0:
        raise ValueError("assumed close must be finite and positive")
    if shares is not None and (not isfinite(shares) or shares <= 0):
        raise ValueError("shares must be finite and positive")
    if next_expected_session(session) is None:
        raise ValueError("scenario date is not a verifiable XNYS session")
    if (exit_today_cost_per_share is None) != (exit_next_open_cost_per_share is None):
        raise ValueError("both exit cost assumptions must be supplied together")
    for cost in (exit_today_cost_per_share, exit_next_open_cost_per_share):
        if cost is not None and (not isfinite(cost) or cost < 0):
            raise ValueError("exit costs must be finite and nonnegative")
    resolved_step = step if step is not None else assumed_close * 0.005
    if not isfinite(resolved_step) or not 0 < resolved_step < assumed_close:
        raise ValueError("scenario step must be positive and smaller than assumed close")
    if any(getattr(model, name, None) is None for name in (
        "day_fit_model", "day_rank_model", "day_model_trained_through",
    )):
        raise ValueError("overnight artifact does not pin its same-day serving models")
    variants = getattr(model, "day_variant_models", None)
    if model.feature_columns == STACKED_FEATURE_COLUMNS and (
        set(variants or {}) != set(VARIANT_OUTPUT_COLUMNS)
        or any(variants[column] is None for column in VARIANT_OUTPUT_COLUMNS)
    ):
        raise ValueError("overnight artifact does not pin its named morning variants")
    if model.day_model_trained_through >= session:
        raise ValueError("same-day serving models saw the scenario session")
    if model.feature_columns == STACKED_FEATURE_COLUMNS and (
        getattr(model, "day_variant_trained_through", None) is None
        or model.day_variant_trained_through >= session
    ):
        raise ValueError("morning variants saw the scenario session")
    if model.label_observed_on > session:
        raise ValueError("overnight model saw a future label")
    if current.corporate_action != "verified_none":
        raise ValueError("current session has a split or dividend")
    if verified.history.empty or not isclose(
        float(verified.history.iloc[-1]["Close"]), current.previous_close,
        rel_tol=1e-4, abs_tol=1e-4,
    ):
        raise ValueError("snapshot previous close differs from verified raw history")
    if len(day_row) != 1:
        raise ValueError("one open-known same-day feature row is required")
    day_outputs = (
        score_day_models(day_row, model.day_fit_model, model.day_rank_model, variants)
        if model.feature_columns == STACKED_FEATURE_COLUMNS else
        score_day_models(day_row, model.day_fit_model, model.day_rank_model)
    )
    provenance = CurrentOpenProvenance("massive", "raw", "massive_actions", current.corporate_action)
    cases = []
    for label, price in (("primary", assumed_close), ("lower", assumed_close - resolved_step), ("higher", assumed_close + resolved_step)):
        scenario = build_scenario_features(
            verified.history, verified.provenance, session=session,
            today_open=current.open, current_open_provenance=provenance,
            assumed_close=price, assumed_close_basis="raw", contract=RAW_CONTRACT,
        )
        forecast = forecast_assumed_close(scenario, day_outputs, model, session=session, contract=RAW_CONTRACT)
        after_cost = None
        if exit_today_cost_per_share is not None and exit_next_open_cost_per_share is not None:
            after_cost = forecast.difference_per_share + exit_today_cost_per_share - exit_next_open_cost_per_share
        cases.append(ForecastCase(label, forecast, scenario.features.to_dict(), after_cost))
    return OvernightScenarioResult(
        ticker, session, current.open, current.last_trade, current.last_trade_at,
        current.fetched_at, resolved_step, shares, tuple(cases), day_outputs,
        model.trained_through, model.label_observed_on, model.feature_version,
        sum(fold.n_rows for fold in model.folds),
    )


def fetch_forecast_inputs(
    ticker: str, session: date,
    *, client: MassiveOvernightClient | None = None,
    feature_store: FeatureStore | None = None,
    price_store: PriceStore | None = None,
    require_prior_session_snapshot: bool = False,
) -> tuple[VerifiedOvernightBars, VerifiedCurrentOpen, pd.DataFrame]:
    """Provider/storage adapter. No legacy daily bar enters the overnight label.

    The stacked contract trains on yesterday's snapshot fields, so it requires
    that exact prior XNYS session. Old 15-feature artifacts keep their existing
    feature-snapshot freshness policy.
    """
    provider = client or MassiveOvernightClient()
    verified = provider.fetch(ticker, session - timedelta(days=40), session - timedelta(days=1))
    current = provider.fetch_current_open(ticker, session)
    spy = provider.fetch_current_open("SPY", session)
    prepared = prepare_one(
        ticker,
        {ticker: {"open": current.open, "last": current.last_trade or current.open,
                  "prev_close": current.previous_close}},
        set(), feature_store or FeatureStore(), price_store or PriceStore(),
        session, spy.open, spy.previous_close, blocked=set(),
    )
    if prepared.row is None:
        raise ValueError(f"same-day feature row unavailable: {prepared.skipped}")
    if require_prior_session_snapshot:
        calendar = xcals.get_calendar("XNYS")
        expected = pd.Timestamp(calendar.previous_session(session.isoformat())).date()
        if prepared.snapshot_date != expected.isoformat():
            raise ValueError(
                f"stacked overnight forecast needs the prior XNYS feature snapshot "
                f"{expected}; found {prepared.snapshot_date or 'none'}"
            )
    return verified, current, prepared.row


def serve_overnight_forecast(
    *, ticker: str, session: date, assumed_close: float,
    step: float | None = None, shares: float | None = None,
    exit_today_cost_per_share: float | None = None,
    exit_next_open_cost_per_share: float | None = None,
    model_store: ModelStore | None = None,
    client: MassiveOvernightClient | None = None,
    feature_store: FeatureStore | None = None,
    price_store: PriceStore | None = None,
) -> OvernightScenarioResult:
    """One explicit request; never retrain or fall back to an unverified bar."""
    if not isfinite(assumed_close) or assumed_close <= 0:
        raise ValueError("assumed close must be finite and positive")
    if shares is not None and (not isfinite(shares) or shares <= 0):
        raise ValueError("shares must be finite and positive")
    if next_expected_session(session) is None:
        raise ValueError("scenario date is not a verifiable XNYS session")
    store = model_store or ModelStore()
    if not store.exists(MODEL_NAME):
        raise FileNotFoundError("no saved overnight model is selected")
    model = store.read(MODEL_NAME)
    if model_summary(model)["serving_inputs_pinned"] is not True:
        raise ValueError("saved overnight model lacks pinned serving inputs")
    if (
        model.label_observed_on > session or model.day_model_trained_through >= session
        or (model.feature_columns == STACKED_FEATURE_COLUMNS and model.day_variant_trained_through >= session)
    ):
        raise ValueError("saved model has seen the scenario session or a later date")
    verified, current, day_row = fetch_forecast_inputs(
        ticker, session, client=client, feature_store=feature_store, price_store=price_store,
        require_prior_session_snapshot=model.feature_columns == STACKED_FEATURE_COLUMNS,
    )
    return build_forecast_cases(
        ticker=ticker, session=session, verified=verified, current=current,
        day_row=day_row, model=model, assumed_close=assumed_close,
        step=step, shares=shares,
        exit_today_cost_per_share=exit_today_cost_per_share,
        exit_next_open_cost_per_share=exit_next_open_cost_per_share,
    )


def serialize_result(result: OvernightScenarioResult) -> dict[str, object]:
    """JSON-friendly, stable result for the API and eventual What if replay."""
    return {
        "ticker": result.ticker, "session": result.session,
        "today_open": result.today_open, "last_trade": result.last_trade,
        "last_trade_at": result.last_trade_at,
        "quote_fetched_at": result.quote_fetched_at,
        "step": result.step, "shares": result.shares,
        "cases": [
            {
                "label": case.label, **asdict(case.forecast),
                "after_cost_difference_per_share": case.after_cost_difference_per_share,
                "gross_difference_for_shares": (
                    case.forecast.difference_per_share * result.shares if result.shares is not None else None
                ),
                "after_cost_difference_for_shares": (
                    case.after_cost_difference_per_share * result.shares
                    if result.shares is not None and case.after_cost_difference_per_share is not None else None
                ),
                "features": case.features,
            }
            for case in result.cases
        ],
        "day_outputs": result.day_outputs.as_features(),
        "model_trained_through": result.model_trained_through,
        "model_label_observed_on": result.model_label_observed_on,
        "model_feature_version": result.model_feature_version,
        "evaluated_rows": result.evaluated_rows,
    }
