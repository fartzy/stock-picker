"""Production direction-margin OOF alignment and leakage checks."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.training import direction_stack as stack
from stock_picker.training.dataset import LABEL_COLUMN


def _pooled_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=40)
    rows = [
        {
            "ticker": ticker,
            "date": date,
            "signal": float(day + ticker_number / 10),
            LABEL_COLUMN: 0.01 if day % 2 else -0.01,
        }
        for ticker_number, ticker in enumerate(("AAA", "BBB"))
        for day, date in enumerate(dates)
    ]
    # Nonconsecutive indices catch accidental positional joins after filtering.
    return pd.DataFrame(rows, index=range(1000, 1000 + 3 * len(rows), 3))


def test_folds_fit_only_on_earlier_dates_and_align_original_rows(monkeypatch):
    pooled = _pooled_frame()
    fits = []
    scores = []

    def fake_fit(frame, params=None, excluded_features=None, included_features=None):
        assert "svc_direction_margin" not in frame.columns
        assert excluded_features == {"unused"}
        assert included_features == {"signal"}
        fit_index = frozenset(frame.index)
        fit_end = frame["date"].max()
        fits.append((fit_index, fit_end))
        return SimpleNamespace(
            model_type="svc_direction",
            feature_names=["signal"],
            fit_index=fit_index,
            fit_end=fit_end,
        )

    def fake_decision_scores(svc, frame):
        assert svc.fit_end < frame["date"].min()
        assert svc.fit_index.isdisjoint(frame.index)
        scores.append((svc.fit_end, frame.index.copy()))
        return frame["signal"].to_numpy() + svc.fit_end.day / 100

    monkeypatch.setattr(stack, "train_svc_direction", fake_fit)
    monkeypatch.setattr(stack, "decision_scores", fake_decision_scores)

    folds = list(stack.iter_direction_margin_folds(
        pooled, excluded_features={"unused"}, included_features={"signal"}
    ))

    assert len(folds) == 4
    assert len(fits) == len(scores) == 3 + 4
    first_outer_train = pooled[pooled["date"] <= fits[3][1]]
    warmup_dates = sorted(first_outer_train["date"].unique())[:2]
    expected_eligible = first_outer_train[~first_outer_train["date"].isin(warmup_dates)]
    assert len(folds[0].baseline_train) == len(expected_eligible)
    assert set(folds[0].baseline_train.index) == set(expected_eligible.index)

    for fold in folds:
        baseline = fold.baseline_train
        stacked = fold.stacked_train
        test = fold.stacked_test
        assert fold.direction_svc.model_type == "svc_direction"
        assert stacked.drop(columns=stack.DIRECTION_MARGIN_COLUMN).equals(baseline)
        assert stacked.index.equals(baseline.index)
        assert test.drop(columns=stack.DIRECTION_MARGIN_COLUMN).equals(
            pooled.loc[test.index]
        )
        assert baseline["date"].max() < test["date"].min()
        assert np.isfinite(stacked[stack.DIRECTION_MARGIN_COLUMN]).all()
        assert np.isfinite(test[stack.DIRECTION_MARGIN_COLUMN]).all()
        assert baseline.index.is_unique and test.index.is_unique
        assert set(baseline.index).isdisjoint(test.index)

    # Every outer test block becomes eligible OOF history in the next fold.
    for earlier, later in zip(folds, folds[1:]):
        assert set(earlier.stacked_test.index).issubset(later.stacked_train.index)


def test_score_direction_margin_rejects_missing_inputs_and_nonfinite_values(monkeypatch):
    frame = pd.DataFrame({"signal": [1.0, 2.0]}, index=[17, 3])
    svc = SimpleNamespace(model_type="svc_direction", feature_names=["signal"])
    monkeypatch.setattr(stack, "decision_scores", lambda *_: np.array([0.4, -0.2]))
    margin = stack.score_direction_margin(frame, svc)
    assert margin.index.tolist() == [17, 3]
    assert margin.tolist() == [0.4, -0.2]

    with pytest.raises(ValueError, match="inputs are missing"):
        stack.score_direction_margin(frame.drop(columns="signal"), svc)
    with pytest.raises(ValueError, match="own margin"):
        stack.score_direction_margin(
            frame.assign(svc_direction_margin=0.1),
            SimpleNamespace(
                model_type="svc_direction", feature_names=["svc_direction_margin"]
            ),
        )
    monkeypatch.setattr(stack, "decision_scores", lambda *_: np.array([np.nan, 0.2]))
    with pytest.raises(ValueError, match="finite and row-aligned"):
        stack.score_direction_margin(frame, svc)
    monkeypatch.setattr(stack, "decision_scores", lambda *_: np.array([0.2]))
    with pytest.raises(ValueError, match="finite and row-aligned"):
        stack.score_direction_margin(frame, svc)


def test_real_direction_svc_produces_finite_oof_margins():
    pooled = _pooled_frame()
    folds = list(stack.iter_direction_margin_folds(
        pooled, included_features={"signal"}
    ))
    assert len(folds) == 4
    for fold in folds:
        assert fold.direction_svc.feature_names == ["signal"]
        assert np.isfinite(fold.stacked_train[stack.DIRECTION_MARGIN_COLUMN]).all()
        assert np.isfinite(fold.stacked_test[stack.DIRECTION_MARGIN_COLUMN]).all()


def test_fit_requires_strictly_earlier_dates(monkeypatch):
    frame = _pooled_frame()
    first_date = frame["date"].min()
    same_day = frame[frame["date"] == first_date]
    monkeypatch.setattr(stack, "train_svc_direction", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="strictly earlier"):
        stack._fit_and_score(same_day.iloc[:1], same_day.iloc[1:], None, None)
    with pytest.raises(ValueError, match="strictly earlier"):
        stack._fit_and_score(same_day.iloc[0:0], same_day, None, None)


def test_oof_rejects_ambiguous_or_persisted_input_rows():
    pooled = _pooled_frame()
    with pytest.raises(ValueError, match="indices must be unique"):
        list(stack.iter_direction_margin_folds(pooled.set_axis([0] * len(pooled))))
    with pytest.raises(ValueError, match="persisted input"):
        list(stack.iter_direction_margin_folds(pooled.assign(svc_direction_margin=0.0)))
