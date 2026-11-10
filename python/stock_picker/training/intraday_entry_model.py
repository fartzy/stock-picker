"""Small, separately validated current-entry → cash-close models.

Whole sessions are split chronologically into fit/calibration/test blocks.
The persisted estimators are exactly those evaluated on the final block:
neither that block nor the calibration labels are used to refit the trees.
Repeated snapshots receive weights that give each session equal influence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

from stock_picker.features.intraday_entry import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS

MODEL_NAME = "intraday_entry"
MODEL_VERSION = "1.0"
LOSS_THRESHOLD = -0.02
MIN_SESSIONS = 60
TREE_OPTIONS = dict(n_estimators=80, max_depth=3, num_leaves=7,
                    min_child_samples=40, reg_lambda=5.0, learning_rate=0.04,
                    n_jobs=2, random_state=1729, verbosity=-1)


def session_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal dates, equal ticker-days within date, equal snapshots within ticker-day."""
    snapshots = frame.groupby(["session", "ticker"])["ticker"].transform("size")
    names = frame.groupby("session")["ticker"].transform("nunique")
    weights = 1.0 / (snapshots.to_numpy(float) * names.to_numpy(float))
    return weights / weights.mean()


def _mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class ProbabilityHead:
    estimator: Any | None
    constant: float
    calibrator: Any | None = None

    def raw(self, x: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            return np.full(len(x), self.constant)
        return self.estimator.predict_proba(x)[:, 1]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        p = self.raw(x)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(_logit(p))[:, 1]
        return np.clip(p, 0.0, 1.0)


def _fit_head(x: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
              calibration_x: pd.DataFrame, calibration_y: np.ndarray,
              calibration_weights: np.ndarray) -> tuple[ProbabilityHead, dict]:
    rate = _mean(y, weights) if len(y) else 0.0
    head = ProbabilityHead(None, rate)
    detail = {"fit_rows": len(y), "fit_positives": int(y.sum()),
              "calibration_rows": len(calibration_y),
              "calibration_positives": int(calibration_y.sum()),
              "fallback": None, "calibration": "sigmoid"}
    if len(np.unique(y)) == 2:
        head.estimator = LGBMClassifier(objective="binary", **TREE_OPTIONS)
        head.estimator.fit(x, y, sample_weight=weights)
    else:
        detail["fallback"] = "Training subset has one or no classes; training empirical rate."
    if len(np.unique(calibration_y)) == 2 and head.estimator is not None:
        # A separate chronological block fits only this sigmoid, not the trees.
        head.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        head.calibrator.fit(_logit(head.raw(calibration_x)), calibration_y,
                            sample_weight=calibration_weights)
    else:
        detail["calibration"] = "unavailable: calibration subset lacks both classes or trees are constant"
    return head, detail


@dataclass
class LogisticBenchmark:
    columns: tuple[str, ...]
    medians: np.ndarray
    scales: np.ndarray
    estimator: Any | None
    constant: float

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.columns].to_numpy(float)
        x = np.where(np.isfinite(x), x, self.medians)
        return (x - self.medians) / self.scales

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            return np.full(len(frame), self.constant)
        return self.estimator.predict_proba(self.matrix(frame))[:, 1]


@dataclass
class IntradayEntryModel:
    columns: tuple[str, ...]
    profit: ProbabilityHead
    conditional_loss: ProbabilityHead
    returns: Any
    logistic: LogisticBenchmark
    metadata: dict

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        x = frame.reindex(columns=self.columns).replace([np.inf, -np.inf], np.nan)
        probability = self.profit.predict(x)
        return {"probability_profit": probability,
                "expected_return": self.returns.predict(x),
                "probability_loss_2pct": (1 - probability) * self.conditional_loss.predict(x)}


def _probability_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {"brier": _mean((y - p) ** 2, weights),
            "log_loss": -_mean(y * np.log(p) + (1-y) * np.log(1-p), weights)}


def _clock(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["observed_at"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")


def _evaluate(frame: pd.DataFrame, predicted: dict, baseline: np.ndarray,
              logistic: np.ndarray, baseline_return: np.ndarray) -> dict:
    if frame.empty:
        return {"rows": 0, "sessions": 0, "stock_days": 0, "brier": None,
                "baseline_brier": None, "logistic_brier": None, "return_mae": None}
    w = session_weights(frame)
    r = frame.target_return.to_numpy(float)
    y = (r > 0).astype(int)
    result = _probability_metrics(y, predicted["probability_profit"], w)
    result.update(rows=len(frame), sessions=int(frame.session.nunique()),
                  stock_days=int(frame[["session", "ticker"]].drop_duplicates().shape[0]),
                  positive_rows=int(y.sum()), loss_2pct_rows=int((r < LOSS_THRESHOLD).sum()),
                  observed_profit_rate=_mean(y, w),
                  baseline_brier=_probability_metrics(y, baseline, w)["brier"],
                  baseline_log_loss=_probability_metrics(y, baseline, w)["log_loss"],
                  logistic_brier=_probability_metrics(y, logistic, w)["brier"],
                  logistic_log_loss=_probability_metrics(y, logistic, w)["log_loss"],
                  return_mae=_mean(np.abs(r-predicted["expected_return"]), w),
                  return_rmse=float(np.sqrt(_mean((r-predicted["expected_return"])**2, w))),
                  baseline_return_mae=_mean(np.abs(r-baseline_return), w),
                  downside_brier=_mean(((r < LOSS_THRESHOLD)-predicted["probability_loss_2pct"])**2, w))
    bins = []
    p = predicted["probability_profit"]
    for lower, upper in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
        mask = (p >= lower) & (p < upper if upper < .999 else p <= 1)
        if mask.any():
            bins.append({"lower": float(lower), "upper": float(min(upper, 1)),
                         "count": int(mask.sum()), "sessions": int(frame.loc[mask, "session"].nunique()),
                         "predicted": _mean(p[mask], w[mask]), "observed": _mean(y[mask], w[mask])})
    result["reliability"] = bins
    return result


def train_intraday_entry(frame: pd.DataFrame, dataset_metadata: dict | None = None) -> IntradayEntryModel:
    """Fit a bounded helper and preserve its untouched final-session evaluation."""
    frame = frame.copy().sort_values(["session", "ticker", "observed_at"]).reset_index(drop=True)
    if not np.isfinite(frame.target_return.to_numpy(float)).all():
        raise ValueError("Entry targets must be finite observed returns.")
    if frame.duplicated(["session", "ticker", "observed_at"]).any():
        raise ValueError("Duplicate stock/checkpoint observations cannot be counted twice.")
    sessions = sorted(frame.session.unique())
    if len(sessions) < MIN_SESSIONS:
        raise ValueError(f"At least {MIN_SESSIONS} completed sessions are required.")
    fit_end, calibration_end = int(len(sessions)*.6), int(len(sessions)*.8)
    fit = frame[frame.session.isin(sessions[:fit_end])]
    calibration = frame[frame.session.isin(sessions[fit_end:calibration_end])]
    test = frame[frame.session.isin(sessions[calibration_end:])]
    columns = tuple(name for name in FEATURE_COLUMNS if name in fit and
                    fit[name].replace([np.inf, -np.inf], np.nan).nunique(dropna=True) > 1)
    if not columns:
        raise ValueError("No varying finite training features are available.")
    x = fit.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    cx = calibration.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    weights, cw = session_weights(fit), session_weights(calibration)
    target = fit.target_return.to_numpy(float)
    ct = calibration.target_return.to_numpy(float)
    profit, profit_detail = _fit_head(x, (target > 0).astype(int), weights,
                                      cx, (ct > 0).astype(int), cw)
    negative, cn = target <= 0, ct <= 0
    conditional, loss_detail = _fit_head(x.loc[negative], (target[negative] < LOSS_THRESHOLD).astype(int),
                                        weights[negative], cx.loc[cn], (ct[cn] < LOSS_THRESHOLD).astype(int), cw[cn])
    returns = LGBMRegressor(objective="regression_l2", **TREE_OPTIONS)
    returns.fit(x, target, sample_weight=weights)
    medians = np.nanmedian(x.to_numpy(float), axis=0)
    scales = np.nanstd(x.to_numpy(float), axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1
    benchmark = LogisticBenchmark(columns, medians, scales, None, _mean((target>0).astype(int), weights))
    if len(np.unique(target > 0)) == 2:
        benchmark.estimator = LogisticRegression(C=.1, max_iter=1000, solver="lbfgs")
        benchmark.estimator.fit(benchmark.matrix(fit), target > 0, sample_weight=weights)
    model = IntradayEntryModel(columns, profit, conditional, returns, benchmark, {})
    clock = _clock(fit)
    clock_rates, clock_returns = {}, {}
    for value in sorted(clock.unique()):
        mask = (clock == value).to_numpy()
        clock_rates[value] = _mean((target[mask] > 0).astype(int), weights[mask])
        clock_returns[value] = _mean(target[mask], weights[mask])
    baseline = _clock(test).map(clock_rates).fillna(benchmark.constant).to_numpy(float)
    baseline_return = _clock(test).map(clock_returns).fillna(_mean(target, weights)).to_numpy(float)
    prediction, logistic = model.predict(test), benchmark.predict(test)
    validation = _evaluate(test, prediction, baseline, logistic, baseline_return)
    rank1 = (test["rank"] == 1).to_numpy()
    validation["rank1"] = _evaluate(test.loc[rank1], {k:v[rank1] for k,v in prediction.items()},
                                    baseline[rank1], logistic[rank1], baseline_return[rank1])
    validation["lower_ranks"] = _evaluate(test.loc[~rank1], {k:v[~rank1] for k,v in prediction.items()},
                                         baseline[~rank1], logistic[~rank1], baseline_return[~rank1])
    validation.update(train_start=sessions[0], train_end=sessions[fit_end-1],
                      calibration_start=sessions[fit_end], calibration_end=sessions[calibration_end-1],
                      test_start=sessions[calibration_end], test_end=sessions[-1],
                      train_rows=len(fit), calibration_rows=len(calibration), test_rows=len(test),
                      test_sessions=len(sessions)-calibration_end,
                      train_sessions=fit_end, calibration_sessions=calibration_end-fit_end,
                      probability_head=profit_detail, conditional_loss_head=loss_detail,
                      training_profit_rate_by_clock=clock_rates,
                      training_return_by_clock=clock_returns)
    gains = np.zeros(len(columns))
    # Equal normalized gain contribution from every fitted tree head, including
    # conditional downside. Importances describe usage, not causal evidence.
    for estimator in (returns, profit.estimator, conditional.estimator):
        if estimator is not None:
            head_gains = estimator.booster_.feature_importance(importance_type="gain").astype(float)
            gains += head_gains / max(head_gains.sum(), 1e-12)
    gains /= max(gains.sum(), 1e-12)
    primary = validation["rank1"]
    better = primary.get("brier") is not None and primary["brier"] < min(primary["baseline_brier"], primary["logistic_brier"])
    model.metadata = {"ready": True, "model_name": MODEL_NAME, "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(), "trained_through": sessions[calibration_end-1],
        "fit_through": sessions[fit_end-1], "evaluated_through": sessions[-1],
        "evidence_status": "experimental", "feature_count": len(columns),
        "features": [{"name": name, "description": FEATURE_DESCRIPTIONS[name], "importance": float(gain)}
                     for name, gain in sorted(zip(columns, gains), key=lambda item: -item[1])],
        "pruned_features": [{"name": name, "reason": "All missing or constant in fitting sessions."}
                            for name in FEATURE_COLUMNS if name not in columns],
        "dataset": dataset_metadata or {}, "validation": validation,
        "notes": ["Experimental: predicts price change from the observed entry to today's cash close, with zero assumed expenses.",
                  "Profit means a close above entry. Close loss >2% means a terminal loss, not touching a price during the day.",
                  "Chronological 60% fit / 20% sigmoid calibration / 20% untouched evaluation. The evaluated artifact is saved without refitting.",
                  "Rank 1 is the primary evaluation. Other ranks have sparse, cache-selected historical coverage.",
                  "Historical morning ranks came from a frozen research model. Current production rank/score context may differ; normalized scores do not eliminate this shift.",
                  "The clock baseline and logistic benchmark use fitting sessions only. Repeated checkpoints are weighted within stock-day and date.",
                  "Rank 1 probability error beat both baselines on this holdout; this is limited evidence, not an established trading edge." if better else
                  "Rank 1 probability error did not beat both simple baselines. Treat these odds as experimental, not an established improvement."]}
    return model
