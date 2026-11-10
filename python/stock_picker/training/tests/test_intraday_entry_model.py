from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stock_picker.features.intraday_entry import FEATURE_COLUMNS
from stock_picker.training.intraday_entry_model import session_weights, train_intraday_entry


def history():
    rng = np.random.default_rng(42)
    rows = []
    for day in range(80):
        session = (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)).date().isoformat()
        for ticker, rank in (("A", 1), ("B", 2)):
            for minute in (40, 55):
                row = {name: float(rng.normal()) for name in FEATURE_COLUMNS}
                row.update(session=session, ticker=ticker, rank=rank,
                           observed_at=f"{session}T14:{minute}:00+00:00",
                           entry_price=100., target_return=float(rng.normal(0, .035)))
                rows.append(row)
    return pd.DataFrame(rows)


def test_whole_dates_are_disjoint_and_test_labels_cannot_change_fit():
    frame = history()
    frame["previous_close_known"] = 1.0
    model = train_intraday_entry(frame, {"source": "synthetic"})
    validation = model.metadata["validation"]
    assert validation["train_end"] < validation["calibration_start"]
    assert validation["calibration_end"] < validation["test_start"]
    assert model.metadata["trained_through"] == validation["calibration_end"]
    assert model.metadata["evaluated_through"] == validation["test_end"]
    assert validation["rank1"]["sessions"] == validation["test_sessions"]
    assert "previous_close_known" not in model.columns
    changed = frame.copy()
    changed.loc[changed.session >= validation["test_start"], "target_return"] += .5
    other = train_intraday_entry(changed)
    predictions, repeated = model.predict(frame), other.predict(frame)
    for key in predictions:
        np.testing.assert_allclose(predictions[key], repeated[key])
    assert model.returns.objective == "regression_l2"
    assert np.all(predictions["probability_profit"] + predictions["probability_loss_2pct"] <= 1 + 1e-12)
    assert validation["baseline_brier"] >= 0
    assert validation["logistic_brier"] >= 0
    assert validation["rank1"]["reliability"]


def test_repeated_checkpoints_share_stock_day_and_session_weight():
    frame = pd.DataFrame({"session": ["a", "a", "a", "b"], "ticker": ["X", "X", "Y", "Z"]})
    weights = session_weights(frame)
    assert weights[0] == weights[1]
    assert weights[0] + weights[1] == weights[2]
    assert weights[:3].sum() == weights[3]


def test_no_negative_outcomes_has_explicit_constant_downside_fallback():
    frame = history()
    frame["target_return"] = .01
    model = train_intraday_entry(frame)
    output = model.predict(frame)
    assert np.all(output["probability_profit"] == 1)
    assert np.all(output["probability_loss_2pct"] == 0)
    assert model.metadata["validation"]["conditional_loss_head"]["fallback"]


def test_rejects_duplicate_snapshots_and_insufficient_sessions():
    frame = history()
    with pytest.raises(ValueError, match="Duplicate"):
        train_intraday_entry(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="60 completed"):
        train_intraday_entry(frame.iloc[:20])
