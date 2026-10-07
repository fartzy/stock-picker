"""Close-conditioned next-open model, with morning-model outputs as inputs.

Ten columns come from ``overnight_dataset``, including strictly prior
five-session momentum. Four are the same-day Fit, Rank, SVR, and direction-SVC
outputs available after today's open; the last compares Fit's prediction with
the assumed day move. Historical morning values must be scored by models
fitted before their session. Intraday volatility is not replaced by a
completed bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

import lightgbm as lgb
import numpy as np
import pandas as pd

from stock_picker.training.ensemble import Ensemble, predict_ensemble
from stock_picker.training.overnight_dataset import (
    CurrentOpenProvenance,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    PriceContract,
    ScenarioBuild,
    build_scenario_features,
    next_expected_session,
)
from stock_picker.training.splits import walk_forward_splits
from stock_picker.training.svm_stack import score_stacked_svm


MODEL_NAME = "next_open_from_assumed_close"
MODEL_FEATURE_VERSION = "overnight_with_morning_outputs_v2"
DAY_OUTPUT_COLUMNS = (
    "day_fit_predicted_return",
    "day_rank_score",
    "day_svr_predicted_return",
    "day_svc_direction_margin",
)
FIT_RESIDUAL_COLUMN = "fit_minus_assumed_day_return"
MODEL_FEATURE_COLUMNS = (*FEATURE_COLUMNS, *DAY_OUTPUT_COLUMNS, FIT_RESIDUAL_COLUMN)
SVM_OUTPUT_COLUMNS = ("svr_oof_pred", "svc_direction_margin")
assert len(MODEL_FEATURE_COLUMNS) == 15

DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "num_threads": 4,
    "seed": 42,
}
DEFAULT_ROUNDS = 80


@dataclass(frozen=True)
class DayModelOutputs:
    fit_predicted_return: float
    rank_score: float
    svr_predicted_return: float
    svc_direction_margin: float

    def as_features(self) -> dict[str, float]:
        values = dict(zip(DAY_OUTPUT_COLUMNS, (
            self.fit_predicted_return,
            self.rank_score,
            self.svr_predicted_return,
            self.svc_direction_margin,
        )))
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("same-day model outputs must be finite")
        return values


@dataclass(frozen=True)
class OvernightFoldMetrics:
    train_through: date
    test_start: date
    test_end: date
    n_rows: int
    model_gap_mae: float
    zero_gap_mae: float
    ticker_mean_gap_mae: float
    model_open_mae: float
    unchanged_open_mae: float
    ticker_mean_open_mae: float


@dataclass
class OvernightModel:
    booster: lgb.Booster
    contract: dict[str, object]
    feature_columns: tuple[str, ...]
    feature_version: str
    trained_through: date
    label_observed_on: date
    folds: tuple[OvernightFoldMetrics, ...]
    gap_abs_error_p90: float
    open_abs_error_p90: float
    day_model_source: dict[str, object] | None = None


@dataclass(frozen=True)
class OvernightForecast:
    assumed_close: float
    predicted_gap: float
    projected_open: float
    difference_per_share: float
    next_session: date | None
    oof_model_gap_mae: float | None
    oof_zero_gap_mae: float | None
    oof_ticker_mean_gap_mae: float | None
    oof_abs_open_error_p90_at_assumed_price: float | None


def score_day_model_frame(
    open_known_rows: pd.DataFrame, fit: Ensemble, rank: Ensemble,
) -> pd.DataFrame:
    """One train/serve scoring path using the actual saved morning estimators."""
    if open_known_rows.empty:
        raise ValueError("at least one open-known same-day row is required")
    estimators = getattr(fit, "stacked_svm_estimators", None) or {}
    margins = score_stacked_svm(open_known_rows, estimators, SVM_OUTPUT_COLUMNS)
    scored = pd.DataFrame({
        DAY_OUTPUT_COLUMNS[0]: predict_ensemble(fit, open_known_rows),
        DAY_OUTPUT_COLUMNS[1]: predict_ensemble(rank, open_known_rows),
        DAY_OUTPUT_COLUMNS[2]: margins["svr_oof_pred"].to_numpy(),
        DAY_OUTPUT_COLUMNS[3]: margins["svc_direction_margin"].to_numpy(),
    }, index=open_known_rows.index)
    if not np.isfinite(scored.to_numpy(dtype=float)).all():
        raise ValueError("same-day model outputs must be finite")
    return scored


def score_day_models(
    open_known_row: pd.DataFrame, fit: Ensemble, rank: Ensemble,
) -> DayModelOutputs:
    """Use the same scoring transform as historical fold generation."""
    if len(open_known_row) != 1:
        raise ValueError("exactly one open-known same-day row is required")
    scored = score_day_model_frame(open_known_row, fit, rank).iloc[0]
    result = DayModelOutputs(
        fit_predicted_return=float(scored[DAY_OUTPUT_COLUMNS[0]]),
        rank_score=float(scored[DAY_OUTPUT_COLUMNS[1]]),
        svr_predicted_return=float(scored[DAY_OUTPUT_COLUMNS[2]]),
        svc_direction_margin=float(scored[DAY_OUTPUT_COLUMNS[3]]),
    )
    result.as_features()
    return result


def attach_historical_day_scores(
    overnight_rows: pd.DataFrame, day_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join a pooled overnight frame to genuinely prior-trained morning scores.

    Both frames have ``ticker`` and ``date``. Scores also require
    ``trained_through``; accepting a current fitted model's retrospective
    predictions would leak the same-day label into overnight training.
    """
    keys = {"ticker", "date"}
    required = keys | {"trained_through", *DAY_OUTPUT_COLUMNS}
    if keys - set(overnight_rows) or required - set(day_scores):
        raise ValueError("pooled overnight rows or historical day scores lack required columns")
    if overnight_rows.duplicated(["ticker", "date"]).any() or day_scores.duplicated(["ticker", "date"]).any():
        raise ValueError("ticker/date scores must be unique")
    if set(DAY_OUTPUT_COLUMNS) & set(overnight_rows):
        raise ValueError("overnight rows must not contain unverified day outputs")
    joined = overnight_rows.merge(day_scores[list(required)], on=["ticker", "date"], how="left", validate="one_to_one")
    if joined[list(DAY_OUTPUT_COLUMNS)].isna().any().any():
        raise ValueError("every overnight row needs all four historical same-day outputs")
    dates = pd.to_datetime(joined["date"], errors="raise")
    fitted = pd.to_datetime(joined["trained_through"], errors="coerce")
    if (
        dates.isna().any() or dates.dt.tz is not None
        or not dates.equals(dates.dt.normalize())
        or fitted.isna().any() or fitted.dt.tz is not None
        or not (fitted.dt.normalize() < dates).all()
    ):
        raise ValueError("every same-day score must come from a strictly earlier fitted model")
    values = joined[list(DAY_OUTPUT_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("historical same-day outputs must be finite")
    return joined.drop(columns="trained_through")


def add_model_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same assumption-vs-Fit transform to training and scenarios."""
    required = {"assumed_day_return", DAY_OUTPUT_COLUMNS[0]}
    if required - set(frame):
        raise ValueError("overnight frame lacks inputs for the Fit residual")
    residual = frame[DAY_OUTPUT_COLUMNS[0]] - frame["assumed_day_return"]
    if FIT_RESIDUAL_COLUMN in frame and not np.allclose(
        frame[FIT_RESIDUAL_COLUMN].to_numpy(dtype=float),
        residual.to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("supplied Fit residual does not match its source features")
    return frame.assign(**{FIT_RESIDUAL_COLUMN: residual})


def _validated_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", LABEL_COLUMN, *(name for name in MODEL_FEATURE_COLUMNS if name != FIT_RESIDUAL_COLUMN)}
    if required - set(frame):
        raise ValueError(f"overnight training frame lacks {sorted(required - set(frame))}")
    if frame.empty or frame.duplicated(["ticker", "date"]).any():
        raise ValueError("overnight training rows must be nonempty and unique")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.equals(dates.dt.normalize()):
        raise ValueError("overnight dates must be timezone-naive exchange dates")
    frame = add_model_derived_features(frame)
    values = frame[[*MODEL_FEATURE_COLUMNS, LABEL_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("overnight training features and labels must be finite")
    return frame.assign(date=dates).sort_values(["date", "ticker"]).reset_index(drop=True)


def _fit(
    frame: pd.DataFrame, params: dict | None, rounds: int,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> lgb.Booster:
    if rounds < 1:
        raise ValueError("boosting rounds must be positive")
    dataset = lgb.Dataset(frame[list(feature_columns)], label=frame[LABEL_COLUMN])
    return lgb.train({**DEFAULT_PARAMS, **(params or {})}, dataset, num_boost_round=rounds)


def train_overnight_model(
    frame: pd.DataFrame,
    contract: PriceContract,
    *,
    n_splits: int = 4,
    params: dict | None = None,
    rounds: int = DEFAULT_ROUNDS,
) -> OvernightModel:
    """Chronological evaluation, then one separate model fitted on all rows."""
    rows = _validated_training_frame(frame)
    if rows["date"].nunique() < n_splits + 1:
        raise ValueError("not enough distinct sessions for walk-forward evaluation")
    metrics = []
    gap_residuals: list[float] = []
    open_residuals: list[float] = []
    for train_mask, test_mask in walk_forward_splits(rows["date"], n_splits=n_splits):
        train, test = rows.loc[train_mask], rows.loc[test_mask]
        if train.empty or test.empty or train["date"].max() >= test["date"].min():
            raise ValueError("invalid chronological overnight fold")
        booster = _fit(train, params, rounds)
        predicted = np.asarray(booster.predict(test[list(MODEL_FEATURE_COLUMNS)]), dtype=float)
        actual = test[LABEL_COLUMN].to_numpy(dtype=float)
        assumed = test["assumed_close"].to_numpy(dtype=float)
        ticker_means = train.groupby("ticker")[LABEL_COLUMN].mean()
        pooled_mean = float(train[LABEL_COLUMN].mean())
        mean_prediction = test["ticker"].map(ticker_means).fillna(pooled_mean).to_numpy(dtype=float)
        gap_residuals.extend(np.abs(predicted - actual).tolist())
        open_residuals.extend(np.abs(assumed * (predicted - actual)).tolist())
        metrics.append(OvernightFoldMetrics(
            train_through=train["date"].max().date(),
            test_start=test["date"].min().date(),
            test_end=test["date"].max().date(),
            n_rows=len(test),
            model_gap_mae=float(np.mean(np.abs(predicted - actual))),
            zero_gap_mae=float(np.mean(np.abs(actual))),
            ticker_mean_gap_mae=float(np.mean(np.abs(mean_prediction - actual))),
            model_open_mae=float(np.mean(np.abs(assumed * (predicted - actual)))),
            unchanged_open_mae=float(np.mean(np.abs(assumed * actual))),
            ticker_mean_open_mae=float(np.mean(np.abs(assumed * (mean_prediction - actual)))),
        ))
    trained_through = rows["date"].max().date()
    label_observed_on = next_expected_session(trained_through)
    if label_observed_on is None:
        raise ValueError("the latest overnight label has no verifiable next session")
    return OvernightModel(
        booster=_fit(rows, params, rounds),
        contract={
            **contract.artifact_metadata(),
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "base_feature_columns": FEATURE_COLUMNS,
            "feature_version": MODEL_FEATURE_VERSION,
        },
        feature_columns=MODEL_FEATURE_COLUMNS,
        feature_version=MODEL_FEATURE_VERSION,
        trained_through=trained_through,
        label_observed_on=label_observed_on,
        folds=tuple(metrics),
        gap_abs_error_p90=float(np.quantile(gap_residuals, 0.9)),
        open_abs_error_p90=float(np.quantile(open_residuals, 0.9)),
    )


def forecast_assumed_close(
    scenario: ScenarioBuild,
    day_outputs: DayModelOutputs,
    model: OvernightModel,
    *,
    session: date,
    contract: PriceContract,
) -> OvernightForecast:
    """Recompute for each assumed close; never shift an earlier prediction."""
    if scenario.features is None or scenario.exclusion_reason is not None:
        raise ValueError(f"invalid overnight scenario: {scenario.exclusion_reason}")
    if scenario.next_session is None:
        raise ValueError("overnight scenario has no verified next exchange session")
    if model.feature_columns != MODEL_FEATURE_COLUMNS or model.feature_version != MODEL_FEATURE_VERSION:
        raise ValueError("overnight model feature contract is incompatible")
    if model.label_observed_on > session:
        raise ValueError("overnight model has seen the scenario session or a later date")
    if any(model.contract.get(key) != getattr(contract, key) for key in ("source", "basis", "action_source")):
        raise ValueError("overnight model price contract is incompatible")
    row = scenario.features.to_dict() | day_outputs.as_features()
    features = add_model_derived_features(pd.DataFrame([row]))[list(MODEL_FEATURE_COLUMNS)]
    if features.isna().any().any():
        raise ValueError("overnight forecast has missing features")
    gap = float(model.booster.predict(features)[0])
    if not isfinite(gap) or gap <= -1:
        raise ValueError("overnight model returned an invalid next-open gap")
    assumed = float(row["assumed_close"])
    projected = assumed * (1 + gap)
    evaluated = sum(fold.n_rows for fold in model.folds)

    def weighted(name: str) -> float | None:
        return (
            sum(getattr(fold, name) * fold.n_rows for fold in model.folds) / evaluated
            if evaluated else None
        )

    return OvernightForecast(
        assumed_close=assumed,
        predicted_gap=gap,
        projected_open=projected,
        difference_per_share=projected - assumed,
        next_session=scenario.next_session,
        oof_model_gap_mae=weighted("model_gap_mae"),
        oof_zero_gap_mae=weighted("zero_gap_mae"),
        oof_ticker_mean_gap_mae=weighted("ticker_mean_gap_mae"),
        oof_abs_open_error_p90_at_assumed_price=assumed * model.gap_abs_error_p90 if evaluated else None,
    )


def forecast_if_closes_at(
    *,
    prior_history: pd.DataFrame,
    provenance: pd.DataFrame,
    session: date,
    today_open: float,
    current_open_provenance: CurrentOpenProvenance,
    assumed_close: float,
    open_known_day_row: pd.DataFrame,
    fit_model: Ensemble,
    rank_model: Ensemble,
    fit_trained_through: date,
    rank_trained_through: date,
    overnight_model: OvernightModel,
    contract: PriceContract,
) -> OvernightForecast:
    """The full hypothetical-close question, independent of API or UI."""
    if fit_trained_through >= session or rank_trained_through >= session:
        raise ValueError("same-day models must be fitted before the scenario session")
    scenario = build_scenario_features(
        prior_history,
        provenance,
        session=session,
        today_open=today_open,
        current_open_provenance=current_open_provenance,
        assumed_close=assumed_close,
        assumed_close_basis=contract.basis,
        contract=contract,
    )
    if scenario.features is None:
        raise ValueError(f"invalid overnight scenario: {scenario.exclusion_reason}")
    return forecast_assumed_close(
        scenario,
        score_day_models(open_known_day_row, fit_model, rank_model),
        overnight_model,
        session=session,
        contract=contract,
    )
