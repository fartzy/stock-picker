"""Leakage and candidate-selection checks for the SVM stacking search."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training import svr_stack_search as search
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.model import feature_columns
from stock_picker.training.svr_stack_search import (
    _oof_training_views,
    _parse_svm_names,
    _require_earlier_dates,
    candidate_columns,
)


def test_stacked_svm_block_requires_strictly_earlier_fit_dates():
    fitted_on = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05"])})
    scored_later = pd.DataFrame({"date": pd.to_datetime(["2026-01-06", "2026-01-07"])})
    scored_same_day = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"])})

    _require_earlier_dates(fitted_on, scored_later)
    with pytest.raises(ValueError, match="strictly earlier"):
        _require_earlier_dates(fitted_on, scored_same_day)
    with pytest.raises(ValueError, match="strictly earlier"):
        _require_earlier_dates(fitted_on, fitted_on.iloc[0:0])


def test_oof_training_views_keep_baseline_and_stack_on_the_same_rows():
    train_frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]), "signal": [1, 2, 3]},
        index=[10, 11, 12],
    )
    oof_block = train_frame.loc[[12, 11]].assign(
        **{name: [0.2, 0.1] for name in STACKED_SVM_COLUMNS}
    )

    baseline_train, stacked_train = _oof_training_views(train_frame, [oof_block])

    assert baseline_train.index.tolist() == [12, 11]
    assert stacked_train.index.tolist() == baseline_train.index.tolist()
    assert "svr_oof_pred" not in baseline_train.columns
    assert stacked_train["svr_oof_pred"].tolist() == [0.2, 0.1]
    with pytest.raises(ValueError, match="unique subset"):
        _oof_training_views(train_frame, [oof_block, oof_block])
    with pytest.raises(ValueError, match="dates do not match"):
        _oof_training_views(train_frame, [oof_block.assign(date=pd.Timestamp("2026-02-01"))])
    with pytest.raises(ValueError, match="identities, labels or base features"):
        _oof_training_views(train_frame, [oof_block.assign(signal=999)])
    with pytest.raises(ValueError, match="finite"):
        _oof_training_views(train_frame, [oof_block.assign(svr_oof_pred=np.nan)])


def test_candidate_ablation_sets_preserve_named_aliases_and_experiment_only_selection():
    candidates = candidate_columns()
    values = list(candidates.values())
    assert candidates["svr_only"] == candidates["only_svr_oof_pred"]
    assert candidates["drop_svr_oof_pred"] == candidates["svc_planes_only"]
    assert set(values) >= {(name,) for name in STACKED_SVM_COLUMNS}
    assert tuple(STACKED_SVM_COLUMNS) in values
    assert tuple(STACKED_SVM_COLUMNS[1:]) in values
    assert all(
        tuple(other for other in STACKED_SVM_COLUMNS if other != name) in values
        for name in STACKED_SVM_COLUMNS
    )
    assert _parse_svm_names(["svc_gate_margin,svr_oof_pred", "svc_gate_margin"]) == (
        "svr_oof_pred", "svc_gate_margin"
    )
    with pytest.raises(ValueError, match="unknown SVM feature"):
        _parse_svm_names(["wrong_name"])
    with pytest.raises(ValueError, match="unknown SVM feature"):
        candidate_columns(excluded_outputs=("wrong_name",))
    with pytest.raises(ValueError, match="all SVM outputs"):
        candidate_columns(excluded_outputs=STACKED_SVM_COLUMNS)

    subset = candidate_columns(
        excluded_outputs=("svc_gate_margin",),
        retained_outputs=("svr_oof_pred", "svc_gate_margin", "svc_direction_margin"),
    )
    assert set(subset.values()) == {
        ("svr_oof_pred",),
        ("svc_direction_margin",),
        ("svr_oof_pred", "svc_direction_margin"),
    }


def test_main_reuses_chronological_svm_fits_and_compares_same_rows(monkeypatch):
    dates = pd.bdate_range("2026-01-02", periods=40)
    pooled = pd.DataFrame(
        [
            {
                "ticker": ticker, "date": date, "signal": 0.1, "noise": 0.2,
                LABEL_COLUMN: 0.01,
            }
            for ticker in ("AAA", "BBB")
            for date in dates
        ]
    )
    loaded_tickers = []
    fits = []
    lightgbm_calls = []

    monkeypatch.setattr(
        search, "UniverseStore",
        lambda: SimpleNamespace(active_tickers=lambda: ["AAA", "BBB", "HOLD"]),
    )
    monkeypatch.setattr(search, "select_holdout_tickers", lambda tickers: {"HOLD"})
    monkeypatch.setattr(
        search, "_load_pooled", lambda tickers, *_: loaded_tickers.extend(tickers) or pooled
    )
    monkeypatch.setattr(search, "pruned_features", lambda: {"noise", "svr_oof_pred"})

    def fake_fit(frame, excluded_features=None, included_features=None):
        assert excluded_features == {"noise"}
        assert included_features is None  # candidate outputs never alter SVM inputs
        assert set(feature_columns(frame, excluded_features)) == {"signal"}
        fits.append(frame["date"].max())
        return {"fitted_through": frame["date"].max()}

    def fake_attach(frame, estimators):
        assert estimators["fitted_through"] < frame["date"].min()
        return frame.assign(**{name: 0.1 for name in STACKED_SVM_COLUMNS})

    def fake_lightgbm(frame, excluded_features=None, included_features=None):
        names = feature_columns(frame, excluded_features, included_features)
        lightgbm_calls.append((tuple(frame.index), set(names)))
        return SimpleNamespace(feature_names=names)

    monkeypatch.setattr(search, "fit_stacked_svm_estimators", fake_fit)
    monkeypatch.setattr(search, "attach_stacked_svm_columns", fake_attach)
    monkeypatch.setattr(search, "train_lightgbm", fake_lightgbm)
    monkeypatch.setattr(search, "predict", lambda model, frame: np.zeros(len(frame)))
    monkeypatch.setattr(
        search, "_fold_metrics",
        lambda *args: {
            "mae": 0.01, "acc": 0.5, "rank_ic": 0.0,
            "gated_n": 0, "gated_hit": 0.0, "gated_avg": 0.0,
        },
    )

    search.main()

    assert set(loaded_tickers) == {"AAA", "BBB"}  # holdout never reaches a data store
    assert len(fits) == search.OOF_SEED_SPLITS + search.N_SPLITS
    group_size = 1 + len(set(candidate_columns().values()))
    assert len(lightgbm_calls) == search.N_SPLITS * group_size
    for offset in range(0, len(lightgbm_calls), group_size):
        baseline_rows, baseline_names = lightgbm_calls[offset]
        assert baseline_names == {"signal"}
        for candidate_rows, candidate_names in lightgbm_calls[offset + 1:offset + group_size]:
            assert candidate_rows == baseline_rows
            assert candidate_names - baseline_names <= set(STACKED_SVM_COLUMNS)
            assert candidate_names > baseline_names
