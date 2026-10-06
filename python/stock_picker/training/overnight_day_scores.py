"""Historical morning-model outputs for the close-conditioned overnight model.

Never score historical rows with today's saved Fit pickle: that model has seen
their labels. Refit the selected same-day architecture in chronological folds,
then score only each fold's later dates. The resulting values are the actual
Fit/Rank/SVM outputs, with a fitting cutoff on every row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_picker.training.dataset import LABEL_COLUMN as DAY_LABEL_COLUMN
from stock_picker.training.ensemble import ModelSpec, train_ensemble
from stock_picker.training.overnight_model import SVM_OUTPUT_COLUMNS, score_day_model_frame
from stock_picker.training.splits import walk_forward_splits
from stock_picker.training.train import run_walk_forward


def generate_historical_day_scores(
    day_training_frame: pd.DataFrame,
    *,
    fit_specs: list[ModelSpec],
    rank_spec: ModelSpec,
    tracking_dir: Path,
    n_splits: int = 4,
) -> pd.DataFrame:
    """Generate four morning outputs, each using strictly earlier fitting dates.

    ``day_training_frame`` must come from ``dataset.build_pooled_dataset``:
    previous completed-session features plus today's open-known inputs.
    ``fit_specs`` must describe the same model family/selection as the live
    Fit artifact. The existing same-day walk-forward trainer handles its SVM
    stacking; Rank is refitted separately on each earlier-date fold.
    """
    required = {"ticker", "date", DAY_LABEL_COLUMN}
    if required - set(day_training_frame):
        raise ValueError("same-day training frame lacks ticker, date, or label")
    if day_training_frame.empty or day_training_frame.duplicated(["ticker", "date"]).any():
        raise ValueError("same-day training rows must be nonempty and unique")
    if rank_spec.model_type != "lightgbm_rank":
        raise ValueError("rank_spec must train the same-day rank model")
    dates = pd.to_datetime(day_training_frame["date"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.equals(dates.dt.normalize()):
        raise ValueError("same-day dates must be timezone-naive exchange dates")
    day_rows = day_training_frame.assign(date=dates).sort_values(["date", "ticker"]).reset_index(drop=True)
    folds = walk_forward_splits(day_rows["date"], n_splits=n_splits)
    if any(not train_mask.any() or not test_mask.any() for train_mask, test_mask in folds):
        raise ValueError("not enough sessions to generate same-day out-of-fold scores")
    fitted = run_walk_forward(
        day_rows,
        n_splits=n_splits,
        specs=fit_specs,
        tracking_dir=tracking_dir,
        stack_direction_margin=True,
    )
    if len(fitted) != len(folds):
        raise ValueError("same-day folds and fitted models do not align")

    blocks = []
    for (train_mask, test_mask), fit_fold in zip(folds, fitted):
        train, test = day_rows.loc[train_mask], day_rows.loc[test_mask]
        trained_through = train["date"].max()
        if trained_through >= test["date"].min():
            raise ValueError("same-day fold was not trained before its scoring dates")
        fit_model = fit_fold.model
        estimators = getattr(fit_model, "stacked_svm_estimators", None) or {}
        if set(SVM_OUTPUT_COLUMNS) - set(estimators):
            raise ValueError("same-day Fit fold lacks the requested saved SVR/SVC estimators")
        rank_model = train_ensemble(train, [rank_spec])
        block = pd.concat([test[["ticker", "date"]], score_day_model_frame(test, fit_model, rank_model)], axis=1)
        block["trained_through"] = trained_through
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)
