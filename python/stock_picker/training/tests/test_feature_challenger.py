import numpy as np
import pandas as pd
import pytest

from stock_picker.features.pruning import RESEARCH_CANDIDATE_COLUMNS
from stock_picker.training import feature_challenger as challenger
from stock_picker.training.dataset import LABEL_COLUMN
from stock_picker.training.ensemble import Ensemble
from stock_picker.training.model import TrainedModel


def _frame(ticker: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-09-08", periods=16)
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        LABEL_COLUMN: np.linspace(-0.01, 0.01, len(dates)),
        "prior_close": 10.0,
        "market_advance_fraction": 0.5,
        "market_above_sma_20d_fraction": 0.6,
        "market_return_dispersion_1d": 0.02,
    })


def test_challenger_keeps_production_artifact_untouched(monkeypatch, tmp_path):
    train, holdout = _frame("TRAIN"), _frame("HOLDOUT")
    monkeypatch.setattr(
        challenger.UniverseStore, "first_seen_by_ticker",
        lambda self: {"TRAIN": "2026-09-08", "HOLDOUT": "2026-09-08"},
    )
    monkeypatch.setattr(challenger.PrunedFeatureStore, "read", lambda self: set())

    def fake_train(frame, *, included_features, **_):
        return TrainedModel(
            model_type="lightgbm_rank", estimator=None,
            feature_names=sorted(included_features),
        )

    monkeypatch.setattr(challenger, "train_lightgbm_rank", fake_train)
    monkeypatch.setattr(challenger, "predict", lambda _model, frame: frame[LABEL_COLUMN].to_numpy())
    saved = []
    monkeypatch.setattr(challenger.ModelStore, "write", lambda self, name, model: saved.append((name, model)))

    output_dir = tmp_path / "research"
    result = challenger.train_and_save_challenger(
        train, holdout, rounds=2, folds=3, output_dir=output_dir,
    )

    assert result["new_features"] == [
        "market_above_sma_20d_fraction", "market_advance_fraction", "market_return_dispersion_1d",
    ]
    assert result["artifact"] == str(output_dir / "rank_breadth_price_2026-09-29.pkl")
    assert saved[0][0].startswith("rank_breadth_price_")
    assert isinstance(saved[0][1], Ensemble)
    assert saved[0][0] != "day_session_return_rank"


def test_challenger_never_overwrites_an_existing_research_artifact(monkeypatch, tmp_path):
    path = tmp_path / "rank_breadth_price_2026-09-29.pkl"
    path.touch()
    monkeypatch.setattr(
        challenger, "paired_rank_evaluation",
        lambda *_args, **_kwargs: (
            {"start": "2026-09-08", "train_through": "2026-09-29"},
            {"prior_close", "market_advance_fraction"},
        ),
    )

    with pytest.raises(FileExistsError):
        challenger.train_and_save_challenger(_frame("TRAIN"), _frame("HOLDOUT"), output_dir=tmp_path)


def test_challenger_respects_an_explicit_breadth_prune(monkeypatch):
    monkeypatch.setattr(
        challenger.UniverseStore, "first_seen_by_ticker",
        lambda self: {"TRAIN": "2026-09-08", "HOLDOUT": "2026-09-08"},
    )
    monkeypatch.setattr(
        challenger.PrunedFeatureStore, "read",
        lambda self: {"market_advance_fraction"},
    )

    with pytest.raises(ValueError, match="all unpruned breadth_price"):
        challenger.paired_rank_evaluation(_frame("TRAIN"), _frame("HOLDOUT"), rounds=2, folds=3)


def test_all_variant_requires_every_research_column(monkeypatch):
    monkeypatch.setattr(
        challenger.UniverseStore, "first_seen_by_ticker",
        lambda self: {"TRAIN": "2026-09-08", "HOLDOUT": "2026-09-08"},
    )
    monkeypatch.setattr(challenger.PrunedFeatureStore, "read", lambda self: set())
    train, holdout = _frame("TRAIN"), _frame("HOLDOUT")

    with pytest.raises(ValueError, match="all unpruned all"):
        challenger.paired_rank_evaluation(train, holdout, variant="all", rounds=2, folds=3)

    for name in RESEARCH_CANDIDATE_COLUMNS:
        train[name] = 0.5
        holdout[name] = 0.5
    monkeypatch.setattr(
        challenger, "train_lightgbm_rank",
        lambda frame, *, included_features, **_: TrainedModel(
            model_type="lightgbm_rank", estimator=None,
            feature_names=sorted(included_features),
        ),
    )
    monkeypatch.setattr(challenger, "predict", lambda _model, frame: frame[LABEL_COLUMN].to_numpy())

    result, selected = challenger.paired_rank_evaluation(
        train, holdout, variant="all", rounds=2, folds=3,
    )

    assert set(result["new_features"]) == RESEARCH_CANDIDATE_COLUMNS
    assert RESEARCH_CANDIDATE_COLUMNS <= selected


def test_full_scope_trains_on_older_history_but_tests_only_recent_dates(monkeypatch):
    monkeypatch.setattr(
        challenger.UniverseStore, "first_seen_by_ticker",
        lambda self: {"TRAIN": "2026-09-08", "HOLDOUT": "2026-09-08"},
    )
    monkeypatch.setattr(challenger.PrunedFeatureStore, "read", lambda self: set())
    recent = _frame("TRAIN")
    older = recent.head(4).copy()
    older["date"] = pd.bdate_range("2026-08-31", periods=4)
    for name in challenger.CHALLENGER_VARIANTS["breadth_price"]:
        older[name] = np.nan
    train = pd.concat([older, recent], ignore_index=True)
    seen = []

    def fake_train(frame, *, included_features, **_):
        seen.append((frame["date"].min(), frame["date"].max()))
        return TrainedModel(
            model_type="lightgbm_rank", estimator=None,
            feature_names=sorted(included_features),
        )

    monkeypatch.setattr(challenger, "train_lightgbm_rank", fake_train)
    monkeypatch.setattr(challenger, "predict", lambda _model, frame: frame[LABEL_COLUMN].to_numpy())

    result, _ = challenger.paired_rank_evaluation(
        train, _frame("HOLDOUT"), scope="full", rounds=2, folds=3,
    )

    assert result["scope"] == "full"
    assert result["start"][:10] == "2026-08-31"
    assert result["evaluation_start"][:10] == "2026-09-08"
    assert all(start == pd.Timestamp("2026-08-31") for start, _ in seen)
    assert result["folds"]["baseline"][0]["test_from"][:10] == "2026-09-14"
