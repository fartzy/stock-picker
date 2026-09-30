"""Walk-forward bake-off of LightGBM with vs without stacked SVM columns.

ADR 0021: LinearSVR is closed as a blender member (ADR 0020). This asks
whether seven out-of-fold SVM numbers help LightGBM as extra columns:

  svr_oof_pred                 LinearSVR predicted open->close return
  svc_direction_margin         LinearSVC distance to the up/down plane
  svc_gate_margin              LinearSVC distance to the +0.5% buy-gate
  svc_down_gate_margin         LinearSVC distance to the -0.5% tail
  svc_strong_up_margin         LinearSVC distance to the +1% plane
  svc_top_quintile_margin      LinearSVC distance to within-day top 20%
  svc_bottom_quintile_margin   LinearSVC distance to within-day bottom 20%

One LinearSVR (expensive to fit) plus six LinearSVC planes that each ask a
different question. These are research candidates, not currently trained
nightly or scored live. If promoted, the selected SVMs would be fit at night;
extra SVRs are not in this search because each would add ~35-45 minutes to
that retrain for a correlated return forecast.

Each fold fits the seven estimators on TRAIN only and attaches their
outputs to the TEST frame. LightGBM training columns are strictly temporal
OOF too: inner walk-forward fits seed the first fold, and each outer fold's
OOF test columns become training rows for the next fold. Solo and stacked
LightGBM always use the same eligible rows. One SVM fit per seed/outer fold
feeds every ablation. Holdout tickers are never loaded, and the experiment
does not write production pickles or touch PrunedFeatureStore.

Promotion bar (ADR 0021): LightGBM-with-the-stacked-columns beats solo
LightGBM in >=3 of 4 folds on both acc and rank IC. Per-fold lists, not
just means. Holdout is a later, separate step if this clears.

Run: bazelisk run //python/stock_picker/training:svr_stack_search
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from stock_picker.features.pruning import pruned_features
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.log import get_logger
from stock_picker.storage.feature_store import FeatureStore
from stock_picker.storage.price_store import PriceStore
from stock_picker.storage.universe_store import UniverseStore
from stock_picker.training.backtest import rank_ic, simulate_trades
from stock_picker.training.dataset import LABEL_COLUMN, build_pooled_dataset
from stock_picker.training.model import (
    attach_stacked_svm_columns,
    feature_columns,
    fit_stacked_svm_estimators,
    predict,
    train_lightgbm,
)
from stock_picker.training.splits import select_holdout_tickers, walk_forward_splits

logger = get_logger(__name__)

N_SPLITS = 4
OOF_SEED_SPLITS = 3
FIT_THRESHOLD = 0.005


def _parse_svm_names(values: list[str] | None) -> tuple[str, ...]:
    """Accept repeated/comma-separated canonical names; reject silent typos."""
    supplied = [name.strip() for value in values or [] for name in value.split(",")]
    if any(not name for name in supplied):
        raise ValueError("SVM feature names cannot be empty")
    unknown = set(supplied) - set(STACKED_SVM_COLUMNS)
    if unknown:
        raise ValueError(f"unknown SVM feature(s): {', '.join(sorted(unknown))}")
    return tuple(name for name in STACKED_SVM_COLUMNS if name in supplied)


def candidate_columns(
    excluded_outputs: tuple[str, ...] = (),
    retained_outputs: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    """Ablations of research outputs, independent of production base pruning.

    ``retained_outputs`` is an explicit experiment-only subset; exclusions
    are then applied to it. Named aliases are preserved for reporting, while
    the runner fits LightGBM only once per distinct subset per fold. The SVM
    input feature set never changes between these candidates.
    """
    unknown = (set(excluded_outputs) | set(retained_outputs)) - set(STACKED_SVM_COLUMNS)
    if unknown:
        raise ValueError(f"unknown SVM feature(s): {', '.join(sorted(unknown))}")
    available = tuple(
        name for name in STACKED_SVM_COLUMNS
        if (not retained_outputs or name in retained_outputs) and name not in excluded_outputs
    )
    if not available:
        raise ValueError("all SVM outputs were excluded from the experiment")

    proposals = [(f"only_{name}", (name,)) for name in available]
    proposals.extend(
        [
            ("svr_only", tuple(name for name in available if name == "svr_oof_pred")),
            ("svc_planes_only", tuple(name for name in available if name != "svr_oof_pred")),
            (
                "all_seven" if len(available) == len(STACKED_SVM_COLUMNS) else "selected_subset",
                available,
            ),
        ]
    )
    proposals.extend(
        (f"drop_{name}", tuple(other for other in available if other != name))
        for name in available
    )
    candidates = {}
    for name, columns in proposals:
        if columns:
            candidates[name] = columns
    return candidates


def _load_pooled(tickers, price_store, feature_store):
    # Do not import training.main -- that module is both a py_library source
    # and a py_binary, and gazelle cannot resolve the import (same pattern
    # as tune_experiment.py).
    histories, features_by_ticker = {}, {}
    for ticker in tickers:
        try:
            histories[ticker] = price_store.read(ticker)
            features_by_ticker[ticker] = feature_store.read(ticker)
        except FileNotFoundError:
            histories.pop(ticker, None)
    return build_pooled_dataset(histories, features_by_ticker)


def _fold_metrics(pred, actual, dates):
    gated = simulate_trades(pred, actual, threshold=FIT_THRESHOLD)
    return {
        "mae": float(np.mean(np.abs(pred - actual))),
        "acc": float(np.mean(np.sign(pred) == np.sign(actual))),
        "rank_ic": rank_ic(pred, actual, dates),
        "gated_n": gated["n_trades"],
        "gated_hit": gated["hit_rate"],
        "gated_avg": gated["avg_return"],
    }


def _require_earlier_dates(fitted_on: pd.DataFrame, scored: pd.DataFrame) -> None:
    """A stacked block can only use an SVM fitted before its first scored date."""
    if fitted_on.empty or scored.empty or fitted_on["date"].max() >= scored["date"].min():
        raise ValueError("stacked SVM values require strictly earlier fitting dates")


def _oof_training_views(
    train_frame: pd.DataFrame, oof_blocks: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align solo and stacked LightGBM fits on the same historical rows."""
    if not oof_blocks:
        raise ValueError("OOF training requires at least one scored historical block")
    stacked_train = pd.concat(oof_blocks)
    if not stacked_train.index.is_unique or not stacked_train.index.isin(train_frame.index).all():
        raise ValueError("OOF training rows must be a unique subset of the outer training frame")
    baseline_train = train_frame.loc[stacked_train.index]
    missing = set(STACKED_SVM_COLUMNS) - set(stacked_train.columns)
    if missing:
        raise ValueError(f"OOF training rows lack stacked columns: {sorted(missing)}")
    if not baseline_train["date"].equals(stacked_train["date"]):
        raise ValueError("OOF training dates do not match the outer training frame")
    if not baseline_train.equals(stacked_train.reindex(columns=baseline_train.columns)):
        raise ValueError("OOF training identities, labels or base features changed")
    if not np.isfinite(stacked_train[list(STACKED_SVM_COLUMNS)].to_numpy()).all():
        raise ValueError("OOF training SVM outputs must be finite")
    return baseline_train, stacked_train


def main(
    excluded_outputs: tuple[str, ...] = (),
    retained_outputs: tuple[str, ...] = (),
) -> None:
    started = time.time()
    tickers = UniverseStore().active_tickers()
    holdout = select_holdout_tickers(tickers)
    train_tickers = [t for t in tickers if t not in holdout]
    logger.info("train tickers=%s holdout=%s (holdout untouched)", len(train_tickers), len(holdout))
    pooled = _load_pooled(train_tickers, PriceStore(), FeatureStore())
    if set(STACKED_SVM_COLUMNS) & set(pooled.columns):
        raise ValueError("SVM outputs must not appear in the production feature snapshot")
    # Production exclusions apply only to ordinary input features. Which
    # experimental outputs LightGBM receives is a separate research choice.
    excluded = pruned_features() - set(STACKED_SVM_COLUMNS)
    base_features = set(feature_columns(pooled, excluded_features=excluded))
    candidates = candidate_columns(excluded_outputs, retained_outputs)
    logger.info(
        "rows=%s candidates=%s (%.1fs)", len(pooled), list(candidates), time.time() - started
    )

    splits = walk_forward_splits(pooled["date"], n_splits=N_SPLITS)
    oof_training_blocks = []
    # The first outer training block has no earlier outer fold to supply
    # stacked columns. Three smaller chronological fits cover its later
    # 75%; the first quarter is warm-up and is excluded from BOTH
    # LightGBM candidates in every outer fold.
    first_train = pooled[splits[0][0]]
    seed_splits = walk_forward_splits(first_train["date"], n_splits=OOF_SEED_SPLITS)
    for seed_fold, (seed_train_mask, seed_test_mask) in enumerate(seed_splits, start=1):
        seed_train = first_train[seed_train_mask]
        seed_test = first_train[seed_test_mask]
        _require_earlier_dates(seed_train, seed_test)
        logger.info(
            "OOF seed %s/%s fit=%s through %s, score=%s from %s",
            seed_fold,
            OOF_SEED_SPLITS,
            len(seed_train),
            seed_train["date"].max(),
            len(seed_test),
            seed_test["date"].min(),
        )
        seed_estimators = fit_stacked_svm_estimators(seed_train, excluded_features=excluded)
        oof_training_blocks.append(attach_stacked_svm_columns(seed_test, seed_estimators))

    baseline_folds = []
    candidate_folds = {name: [] for name in candidates}
    for fold, (train_mask, test_mask) in enumerate(splits, start=1):
        train_frame = pooled[train_mask]
        test_frame = pooled[test_mask]
        _require_earlier_dates(train_frame, test_frame)
        baseline_train, stacked_train = _oof_training_views(train_frame, oof_training_blocks)
        t0 = time.time()
        logger.info(
            "fold %s/%s train=%s/%s through %s, test=%s from %s",
            fold,
            N_SPLITS,
            len(baseline_train),
            len(train_frame),
            train_frame["date"].max(),
            len(test_frame),
            test_frame["date"].min(),
        )
        baseline = train_lightgbm(baseline_train, excluded_features=excluded)
        if set(baseline.feature_names) != base_features:
            raise ValueError("baseline did not train on the intended production feature set")
        actual = test_frame[LABEL_COLUMN]
        dates = test_frame["date"]
        baseline_metrics = _fold_metrics(
            pd.Series(predict(baseline, test_frame), index=test_frame.index), actual, dates
        )
        baseline_folds.append(baseline_metrics)
        logger.info(
            "fold %s baseline acc=%.6f rank_ic=%.6f gated_n=%s gated_hit=%.6f gated_avg=%.6f (%.1fs)",
            fold,
            baseline_metrics["acc"],
            baseline_metrics["rank_ic"],
            baseline_metrics["gated_n"],
            baseline_metrics["gated_hit"],
            baseline_metrics["gated_avg"],
            time.time() - t0,
        )

        estimators = fit_stacked_svm_estimators(train_frame, excluded_features=excluded)
        t_svm = time.time()
        logger.info(
            "fold %s fitted %s SVM estimators (%.1fs elapsed)",
            fold,
            len(estimators),
            t_svm - t0,
        )

        stacked_test = attach_stacked_svm_columns(test_frame, estimators)
        scored_subsets = {}
        for name, columns in candidates.items():
            if columns in scored_subsets:
                metrics, fitted_as = scored_subsets[columns]
                candidate_folds[name].append(metrics)
                logger.info("fold %s %s = %s (same subset; no additional fit)", fold, name, fitted_as)
                continue
            expected_features = base_features | set(columns)
            stacked = train_lightgbm(
                stacked_train,
                excluded_features=excluded,
                included_features=expected_features,
            )
            if set(stacked.feature_names) != expected_features:
                raise ValueError(f"{name} did not train on its intended feature set")
            metrics = _fold_metrics(
                pd.Series(predict(stacked, stacked_test), index=test_frame.index), actual, dates
            )
            candidate_folds[name].append(metrics)
            scored_subsets[columns] = (metrics, name)
            logger.info(
                "fold %s %s acc=%.6f rank_ic=%.6f gated_n=%s gated_hit=%.6f gated_avg=%.6f (%.1fs total)",
                fold,
                name,
                metrics["acc"],
                metrics["rank_ic"],
                metrics["gated_n"],
                metrics["gated_hit"],
                metrics["gated_avg"],
                time.time() - t0,
            )
        # This test block becomes historical training data next fold;
        # its SVM values were generated from strictly earlier dates.
        oof_training_blocks.append(stacked_test)

    def _summarize(name, folds):
        accs = [f["acc"] for f in folds]
        ics = [f["rank_ic"] for f in folds]
        logger.info(
            "%s mean mae=%.5f acc=%.6f rank_ic=%.6f gated_n=%.1f gated_hit=%.6f gated_avg=%.6f",
            name,
            float(np.mean([f["mae"] for f in folds])),
            float(np.mean(accs)),
            float(np.mean(ics)),
            float(np.mean([f["gated_n"] for f in folds])),
            float(np.nanmean([f["gated_hit"] for f in folds])),
            float(np.nanmean([f["gated_avg"] for f in folds])),
        )
        logger.info(
            "  %s acc by fold: %s | rank_ic by fold: %s",
            name,
            [round(a, 4) for a in accs],
            [round(i, 4) for i in ics],
        )
        return accs, ics

    base_acc, base_ic = _summarize("baseline", baseline_folds)
    for name, folds in candidate_folds.items():
        candidate_acc, candidate_ic = _summarize(name, folds)
        acc_wins = sum(s > b for s, b in zip(candidate_acc, base_acc))
        ic_wins = sum(s > b for s, b in zip(candidate_ic, base_ic))
        logger.info(
            "%s vs baseline: acc wins %s/%s, rank_ic wins %s/%s (bar %s)",
            name,
            acc_wins,
            N_SPLITS,
            ic_wins,
            N_SPLITS,
            "cleared" if acc_wins >= 3 and ic_wins >= 3 else "failed",
        )
    logger.info("stacked columns=%s", STACKED_SVM_COLUMNS)
    logger.info("done %.1fs", time.time() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-svm", action="append", metavar="NAME[,NAME]",
        help="experiment-only SVM outputs to prune",
    )
    parser.add_argument(
        "--retain-svm", action="append", metavar="NAME[,NAME]",
        help="experiment-only SVM outputs to retain",
    )
    args = parser.parse_args()
    try:
        excluded_outputs = _parse_svm_names(args.exclude_svm)
        retained_outputs = _parse_svm_names(args.retain_svm)
        candidate_columns(excluded_outputs, retained_outputs)
    except ValueError as exc:
        parser.error(str(exc))
    main(excluded_outputs=excluded_outputs, retained_outputs=retained_outputs)
