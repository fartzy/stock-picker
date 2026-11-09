from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_picker.ingestion.massive_overnight import ObservedCurrentTrade
from stock_picker.storage.jev_forecast_store import JevForecastStore
from stock_picker.training.jev_assessment import assess_jev, prior_feature_context

NOW = datetime(2026, 11, 9, 16, 0, tzinfo=timezone.utc)


def run(tmp_path, *, now=NOW, quote=None, news=None, **extra):
    def ask(state, questions, **kwargs):
        return {"answers": {name: 0.6 for name in questions}, "model": "jev-test", "tokens": 10, "issue": None,
                "requested_at": now.isoformat(), "completed_at": (now + timedelta(seconds=1)).isoformat()}
    defaults = {
        "ticker": "IONS", "kind": "rank", "forecast_store": JevForecastStore(tmp_path),
        "scan_store": SimpleNamespace(read=lambda *args: {"signals": [{"ticker": "IONS", "predicted_return": 0.4564,
            "open_price": 32.27, "prev_close": 39.38, "news_check": {"status": "degraded", "flag": "event"}}]}),
        "quote_client": SimpleNamespace(fetch_current_trade=lambda *args: quote or ObservedCurrentTrade(28.12, now, now)),
        "news_fetcher": lambda *args: news or {"status": "no_news", "sources": ["massive", "finnhub"],
            "issues": [], "articles": [], "checked_at": now.isoformat()},
        "context_for": lambda *args: ({"return_1d": 0.01}, "2026-11-06", []),
        "ask": ask, "now_fn": lambda: now,
    }
    return assess_jev(**{**defaults, **extra})


def test_entry_target_uses_current_trade_and_explicit_costs_not_open(tmp_path):
    forecast = run(tmp_path, round_trip_cost_per_share=0.02, downside_fraction=0.05)
    assert forecast["target"]["entry_price"] == 28.12
    assert forecast["target"]["entry_basis"] == "observed_trade"
    assert forecast["target"]["downside_price"] == pytest.approx(28.12 * 0.95)
    assert forecast["target"]["round_trip_cost_per_share"] == 0.02
    assert forecast["target"]["barrier_start_at"] == forecast["completed_at"]
    assert set(forecast["answers"]) == {"net_positive_by_close", "downside_before_close"}
    assert forecast["state"]["open_price"] == 32.27
    assert forecast["state"]["model_score_semantics"] == "relative_ranking_score"
    assert forecast["state"]["morning_news_judge"]["status"] == "degraded"
    assert JevForecastStore(tmp_path).read(forecast["id"]) == forecast


def test_user_price_is_an_assumption_with_its_own_timestamp(tmp_path):
    old = ObservedCurrentTrade(28.12, NOW - timedelta(minutes=1), NOW)
    forecast = run(tmp_path, quote=old, entry_price=28.2)
    assert forecast["target"]["entry_price"] == 28.2
    assert forecast["target"]["entry_basis"] == "user_assumption"
    assert forecast["target"]["entry_observed_at"] == NOW.isoformat()
    assert forecast["state"]["observed_trade"]["price"] == 28.12


@pytest.mark.parametrize("quote", [
    ObservedCurrentTrade(28, NOW - timedelta(minutes=6), NOW),
    ObservedCurrentTrade(28, NOW + timedelta(seconds=1), NOW),
    ObservedCurrentTrade(28, NOW, NOW + timedelta(seconds=1)),
    ObservedCurrentTrade(28, NOW.replace(hour=13), NOW),
])
def test_invalid_quote_never_reaches_jev(tmp_path, quote):
    with pytest.raises(ValueError, match="stale, future-dated or outside"):
        run(tmp_path, quote=quote, ask=lambda *args, **kwargs: pytest.fail("invalid evidence was sent"))
    assert JevForecastStore(tmp_path).list_records() == []


@pytest.mark.parametrize("now", [
    NOW.replace(hour=14), NOW.replace(hour=21), datetime(2026, 11, 8, 16, tzinfo=timezone.utc),
    datetime(2026, 11, 26, 16, tzinfo=timezone.utc),
    datetime(2026, 11, 27, 18, tzinfo=timezone.utc),
])
def test_only_open_exchange_sessions_allow_entry(tmp_path, now):
    with pytest.raises(ValueError, match="open regular"):
        run(tmp_path, now=now)


def test_overnight_is_separate_near_close_actual_close_to_next_open(tmp_path):
    early_close_day = datetime(2026, 11, 27, 17, 45, tzinfo=timezone.utc)
    forecast = run(tmp_path, now=early_close_day, mode="overnight")
    assert forecast["question_set_version"] == "overnight_v1"
    assert set(forecast["answers"]) == {"net_positive_overnight"}
    assert forecast["target"]["entry_basis"] == "session_close"
    assert forecast["target"]["entry_price"] is None
    assert forecast["target"]["next_session"] == "2026-11-30"
    assert forecast["target"]["exit_at"].startswith("2026-11-30T14:30")
    with pytest.raises(ValueError, match="final 30 minutes"):
        run(tmp_path, mode="overnight")
    with pytest.raises(ValueError, match="no entry assumption"):
        run(tmp_path, now=early_close_day, mode="overnight", entry_price=28)


def test_future_and_undated_news_are_excluded_and_coverage_remains_degraded(tmp_path):
    news = {"status": "complete", "sources": ["finnhub"], "issues": ["massive_unavailable"],
            "checked_at": NOW.isoformat(), "articles": [
                {"headline": "known", "published_at": (NOW - timedelta(minutes=1)).isoformat()},
                {"headline": "future", "published_at": (NOW + timedelta(seconds=1)).isoformat()},
                {"headline": "undated"},
            ]}
    forecast = run(tmp_path, news=news)
    assert [row["headline"] for row in forecast["state"]["news"]["articles"]] == ["known"]
    assert forecast["coverage"]["news_status"] == "degraded"
    assert "future_article_excluded" in forecast["coverage"]["news_issues"]
    assert "massive_unavailable" in forecast["coverage"]["news_issues"]


def test_features_cannot_include_session_or_future_rows():
    frame = pd.DataFrame({"return_1d": [0.01, 0.9, 0.8]}, index=pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10"]))
    context, snapshot, issues = prior_feature_context("IONS", NOW.date(), SimpleNamespace(read=lambda ticker: frame))
    assert context["return_1d"] == 0.01 and snapshot == "2026-11-06" and issues == []


def test_late_response_is_preserved_but_not_a_usable_forecast(tmp_path):
    def late(state, questions, **kwargs):
        return {"answers": {name: 0.6 for name in questions}, "model": "jev-test", "tokens": 10, "issue": None,
                "requested_at": NOW.isoformat(), "completed_at": NOW.replace(hour=21).isoformat()}
    forecast = run(tmp_path, ask=late)
    assert forecast["issue"] == "session_ended_before_answer"
    assert JevForecastStore(tmp_path).read(forecast["id"]) == forecast


@pytest.mark.parametrize("extra", [{"entry_price": -1}, {"entry_price": True}, {"round_trip_cost_per_share": -1},
                                   {"downside_fraction": 1}, {"downside_fraction": float("nan")}])
def test_invalid_price_cost_and_barrier_are_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        run(tmp_path, **extra)
