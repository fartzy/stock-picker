"""Temporal boundary checks for SVM features used in the stacking search."""

import pandas as pd
import pytest

from stock_picker.training.svr_stack_search import _oof_training_views, _require_earlier_dates


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
    oof_block = train_frame.loc[[12, 11]].assign(svr_oof_pred=[0.2, 0.1])

    baseline_train, stacked_train = _oof_training_views(train_frame, [oof_block])

    assert baseline_train.index.tolist() == [12, 11]
    assert stacked_train.index.tolist() == baseline_train.index.tolist()
    assert "svr_oof_pred" not in baseline_train.columns
    assert stacked_train["svr_oof_pred"].tolist() == [0.2, 0.1]
    with pytest.raises(ValueError, match="unique subset"):
        _oof_training_views(train_frame, [oof_block, oof_block])
    with pytest.raises(ValueError, match="dates do not match"):
        _oof_training_views(train_frame, [oof_block.assign(date=pd.Timestamp("2026-02-01"))])
