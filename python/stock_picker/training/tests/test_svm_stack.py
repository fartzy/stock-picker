"""All-output SVM stacking must stay chronological and row-aligned."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training import svm_stack
from stock_picker.training.dataset import LABEL_COLUMN


def _pooled() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=40)
    rows = [
        {"ticker": ticker, "date": day, "signal": float(i + j), LABEL_COLUMN: float((i % 2) * 2 - 1)}
        for i, day in enumerate(dates)
        for j, ticker in enumerate(("AAA", "BBB", "CCC"))
    ]
    return pd.DataFrame(rows, index=range(100, 100 + 2 * len(rows), 2))


def test_oof_fits_strictly_before_every_scored_block_and_preserves_rows(monkeypatch):
    pooled = _pooled()
    outputs = ("svr_oof_pred", "svc_gate_margin")
    fit_ends = []

    def fake_fit(frame, *, excluded_features, outputs):
        fit_end = frame["date"].max()
        fit_ends.append(fit_end)
        assert excluded_features == set(STACKED_SVM_COLUMNS) | {"pruned"}
        return {name: SimpleNamespace(fit_end=fit_end) for name in outputs}

    def fake_score(frame, estimators, names):
        assert all(model.fit_end < frame["date"].min() for model in estimators.values())
        return pd.DataFrame({name: np.arange(len(frame), dtype=float) for name in names}, index=frame.index)

    monkeypatch.setattr(svm_stack, "fit_stacked_svm_estimators", fake_fit)
    monkeypatch.setattr(svm_stack, "score_stacked_svm", fake_score)
    folds = list(svm_stack.iter_stacked_svm_folds(
        pooled, outputs=outputs, excluded_features={"pruned"}
    ))

    assert len(folds) == 4
    assert len(fit_ends) == 7  # three warm-up OOF fits, then four outer fits
    for fold in folds:
        assert fold.stacked_train.drop(columns=list(outputs)).equals(fold.baseline_train)
        assert fold.stacked_train.index.equals(fold.baseline_train.index)
        assert fold.stacked_test.drop(columns=list(outputs)).equals(pooled.loc[fold.stacked_test.index])
        assert fold.baseline_train["date"].max() < fold.stacked_test["date"].min()
        assert np.isfinite(fold.stacked_train[list(outputs)].to_numpy()).all()
    for earlier, later in zip(folds, folds[1:]):
        assert set(earlier.stacked_test.index).issubset(later.stacked_train.index)


def test_stacked_scorer_rejects_missing_estimator_inputs_and_nonfinite(monkeypatch):
    frame = pd.DataFrame({"signal": [1.0, 2.0]}, index=[9, 3])
    model = SimpleNamespace(model_type="svc_gate", feature_names=["signal"])
    monkeypatch.setattr(svm_stack, "decision_scores", lambda *_: np.array([0.2, -0.3]))
    assert svm_stack.score_stacked_svm(frame, {"svc_gate_margin": model}, ("svc_gate_margin",)).index.tolist() == [9, 3]
    with pytest.raises(ValueError, match="fitted SVM is missing"):
        svm_stack.score_stacked_svm(frame, {}, ("svc_gate_margin",))
    with pytest.raises(ValueError, match="inputs are missing"):
        svm_stack.score_stacked_svm(frame.drop(columns="signal"), {"svc_gate_margin": model}, ("svc_gate_margin",))
    with pytest.raises(ValueError, match="all missing"):
        svm_stack.score_stacked_svm(frame.assign(signal=[np.nan, 2.0]), {"svc_gate_margin": model}, ("svc_gate_margin",))
    monkeypatch.setattr(svm_stack, "decision_scores", lambda *_: np.array([np.nan, 0.2]))
    with pytest.raises(ValueError, match="finite and row-aligned"):
        svm_stack.score_stacked_svm(frame, {"svc_gate_margin": model}, ("svc_gate_margin",))


def test_stacked_scorer_does_not_accept_an_estimator_using_derived_inputs():
    frame = pd.DataFrame({"signal": [1.0], "svc_gate_margin": [10.0]})
    model = SimpleNamespace(model_type="svc_gate", feature_names=["signal", "svc_gate_margin"])
    with pytest.raises(ValueError, match="cannot consume"):
        svm_stack.score_stacked_svm(frame, {"svc_gate_margin": model}, ("svc_gate_margin",))
