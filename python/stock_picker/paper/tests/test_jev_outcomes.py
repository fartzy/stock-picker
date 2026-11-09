"""Decision-time event boundaries, missing coverage, and chronology matter more than rounded scores."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from stock_picker.paper.jev_outcomes import downside_label, forecast_outcome, load_jev_evaluation, summarize_jev
from stock_picker.storage.jev_outcome_store import JevOutcomeStore
from stock_picker.ingestion.massive_overnight import MassiveOvernightError

UTC = timezone.utc
NOW = datetime(2026, 11, 9, 22, tzinfo=UTC)


def record(identity="one", session="2026-11-06", **extra):
    value = {"id": identity, "ticker": "TEST", "session": session, "kind": "rank", "list_rank": 1,
             "score": 0.4, "question_set_version": "entry_v1", "model": "jev-1.14", "issue": None,
             "requested_at": session + "T20:57:00+00:00", "completed_at": session + "T20:58:00+00:00",
             "state": {"news_judge": {"flag": None}}, "answers": {"net_positive_by_close": 0.8, "downside_before_close": 0.1},
             "target": {"entry_price": 10, "entry_basis": "observed_trade", "round_trip_cost_per_share": 0.1,
                        "exit_at": session + "T21:00:00+00:00", "barrier_start_at": session + "T20:58:00+00:00", "downside_price": 9.7}}
    value.update(extra)
    return value


def bar(stamp, low):
    return {"t": int(stamp.timestamp() * 1000), "o": low + .05, "h": low + .1, "l": low, "c": low + .05}


def test_downside_ignores_predecision_low_and_never_fills_missing_minutes():
    start = datetime(2026, 11, 6, 20, 58, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    before = bar(start - timedelta(minutes=1), 8)
    assert downside_label([before, bar(start, 10)], start, end, 9.7)["value"] is None
    assert downside_label([before, bar(start, 10), bar(start + timedelta(minutes=1), 9.9)], start, end, 9.7)["value"] is False
    assert downside_label([bar(start + timedelta(minutes=1), 9.5)], start, end, 9.7)["value"] is True


def test_partial_decision_minute_is_ambiguous_unless_whole_bar_clears_barrier():
    minute = datetime(2026, 11, 6, 20, 58, tzinfo=UTC)
    start, end = minute + timedelta(seconds=30), minute + timedelta(minutes=2)
    later = bar(minute + timedelta(minutes=1), 10)
    assert downside_label([bar(minute, 9.5), later], start, end, 9.7)["value"] is None
    assert downside_label([bar(minute, 9.8), later], start, end, 9.7)["value"] is False


def test_duplicate_minute_cannot_erase_a_touch_regardless_of_order():
    start = datetime(2026, 11, 6, 20, 58, tzinfo=UTC)
    rows = [bar(start, 9.5), bar(start, 10), bar(start + timedelta(minutes=1), 10)]
    assert downside_label(rows, start, start + timedelta(minutes=2), 9.7)["value"] is True
    assert downside_label(list(reversed(rows)), start, start + timedelta(minutes=2), 9.7)["value"] is True
    partial = start + timedelta(seconds=10)
    assert downside_label(rows, partial, start + timedelta(minutes=2), 9.7)["value"] is None


def test_malformed_minute_timestamps_and_ohlc_cannot_prove_no_touch():
    start = datetime(2026, 11, 6, 20, 58, tzinfo=UTC)
    second = bar(start + timedelta(minutes=1), 10)
    malformed = [dict(bar(start, 10), t=start.timestamp() * 1000 + .5),
                 dict(bar(start, 10), h=9), dict(bar(start, 10), c=float("nan")),
                 {"t": start.timestamp() * 1000, "l": 10}]
    for row in malformed:
        assert downside_label([row, second], start, start + timedelta(minutes=2), 9.7)["value"] is None


def test_net_return_costs_pending_and_missing_downside_are_separate():
    forecast = record()
    outcome = {"exit_price": 10.05, "minute_bars": []}
    row = forecast_outcome(forecast, outcome, NOW)
    assert row["net_return"] < 0 and row["labels"]["net_positive_by_close"] is False
    assert row["labels"]["downside_before_close"] is None
    assert forecast_outcome(forecast, outcome, datetime(2026, 11, 6, 20, tzinfo=UTC))["status"] == "pending"
    assert forecast_outcome(record(question_set_version="morning_v1", state={}), outcome, NOW)["status"] == "legacy"


def test_recorded_morning_forecasts_include_untraded_names_and_keep_question_semantics():
    forecast = record(question_set_version="morning_v1", answers={"buy_today": 0.8, "hold_overnight": 0.6, "news_skip": 0.7})
    forecast["target"].update(round_trip_cost_per_share=0, next_session="2026-11-09")
    row = forecast_outcome(forecast, {"exit_price": 10.5, "next_open": 10.7}, NOW)
    assert row["labels"] == {"buy_today": True, "hold_overnight": True}
    summary = summarize_jev([forecast], {"one": {"exit_price": 10.5, "next_open": 10.7}}, now=NOW)
    assert summary["primary_count"] == 1 and summary["counts"]["completed"] == 1
    assert {t["question"] for t in summary["cohorts"][0]["targets"]} == {"buy_today", "hold_overnight"}
    hold = next(t for t in summary["cohorts"][0]["targets"] if t["question"] == "hold_overnight")
    band = next(b for b in hold["reliability"] if b["n"])
    assert abs(band["mean_net_return"] - (10.7 / 10.5 - 1)) < 1e-9


def test_overnight_profit_uses_actual_close_not_previous_last_trade():
    forecast = record(question_set_version="overnight_v1", answers={"net_positive_overnight": 0.6})
    forecast["target"].update(entry_price=None, entry_basis="session_close", exit_at="2026-11-09T14:30:00+00:00")
    row = forecast_outcome(forecast, {"close_price": 11, "exit_price": 10.9}, NOW)
    assert row["labels"]["net_positive_overnight"] is False
    assert abs(row["net_return"] - (-0.2 / 11)) < 1e-9


def test_different_cost_and_barrier_definitions_never_share_a_cohort():
    first, other = record(), record("other")
    other["target"]["downside_fraction"] = 0.05
    result = summarize_jev([first, other], {}, now=NOW)
    assert len(result["cohorts"]) == 2 and result["duplicate_count"] == 0


def test_all_failed_requests_remain_visible_without_inflating_performance_sample():
    result = summarize_jev([record(issue="not_configured", answers={})], {}, now=NOW)
    assert result["failed_count"] == 1 and result["primary_count"] == 0
    assert result["rows"][0]["status"] == "unavailable"


def test_degraded_flag_is_flagged_and_unknown_news_is_not_clear():
    flagged = record("flagged", state={"morning_news_judge": {"status": "degraded", "flag": "dilution"}})
    unknown = record("unknown", ticker="OTHER", state={"morning_news_judge": {"status": "not_checked"}})
    result = summarize_jev([flagged, unknown], {}, now=NOW)
    assert [row["news_group"] for row in result["rows"]] == ["flagged", "not_checked"]


def test_initial_forecasts_deduplicate_and_baseline_uses_strictly_earlier_days():
    first, repeated, next_day = record(), record("repeat"), record("next", "2026-11-09")
    repeated["completed_at"] = "2026-11-06T20:58:30+00:00"
    next_day["answers"]["net_positive_by_close"] = 0.4
    result = summarize_jev([repeated, next_day, first], {"one": {"exit_price": 11}, "repeat": {"exit_price": 11}, "next": {"exit_price": 9}}, now=NOW)
    assert result["primary_count"] == 2 and result["duplicate_count"] == 1
    metric = next(target for target in result["cohorts"][0]["targets"] if target["question"] == "net_positive_by_close")
    assert abs(metric["brier"] - 0.1) < 1e-9
    assert metric["baseline_n"] == 1 and metric["prior_day_baseline_brier"] == 1
    assert abs(metric["matched_jev_brier"] - 0.16) < 1e-9
    assert result["cohorts"][0]["comparisons"][2]["n"] == 2


def test_models_and_questions_stay_separate_and_invalid_probability_not_scored():
    other = record("other", model="new-model")
    other["answers"] = {"net_positive_by_close": 1.1, "downside_before_close": 0.2}
    result = summarize_jev([record(), other], {"one": {"exit_price": 11}, "other": {"exit_price": 11}}, now=NOW)
    assert len(result["cohorts"]) == 2
    new = next(c for c in result["cohorts"] if c["model"] == "new-model")
    assert next(t for t in new["targets"] if t["question"] == "net_positive_by_close")["n"] == 0


class NoTrades:
    def read(self):
        return pd.DataFrame(columns=["ticker", "executed_at"])


def test_read_only_evaluation_never_calls_provider_and_all_candidates_can_refresh(tmp_path):
    class Provider:
        calls = 0
        def fetch_session_close(self, ticker, session):
            self.calls += 1
            return type("Close", (), {"price": 11, "session": session})()
        def fetch_minute_bars(self, ticker, session):
            return []
    provider = Provider()
    store = JevOutcomeStore(data_dir=tmp_path)
    records = [record(), record("another", ticker="UNTRADED")]
    initial = load_jev_evaluation(records, outcome_store=store, provider=provider, trade_store=NoTrades(), now=NOW)
    assert provider.calls == 0 and initial["counts"]["pending"] == 2
    result = load_jev_evaluation(records, refresh=True, outcome_store=store, provider=provider, trade_store=NoTrades(), now=NOW)
    assert provider.calls == 2 and result["counts"]["completed"] == 2
    assert all(not row["traded"] for row in result["rows"])


def test_fully_resolved_outcome_is_not_refetched(tmp_path):
    forecast = record()
    start = datetime(2026, 11, 6, 20, 58, tzinfo=UTC)
    store = JevOutcomeStore(data_dir=tmp_path)
    store.append({"forecast_id": "one", "observed_at": NOW.isoformat(), "exit_price": 11,
                  "minute_bars": [bar(start, 10), bar(start + timedelta(minutes=1), 10)]})
    class FailingProvider:
        def fetch_session_close(self, *_):
            raise AssertionError("A fully resolved outcome must not be fetched again")
    result = load_jev_evaluation([forecast], refresh=True, outcome_store=store, provider=FailingProvider(), trade_store=NoTrades(), now=NOW + timedelta(minutes=1))
    assert result["counts"]["completed"] == 1 and result["rows"][0]["net_return"] == .09


def test_partial_outcome_preserves_known_prices_and_minutes_after_failed_refresh_and_reload(tmp_path):
    start = datetime(2026, 11, 6, 20, 58, tzinfo=UTC)
    store = JevOutcomeStore(data_dir=tmp_path)
    prior = {"forecast_id": "one", "observed_at": NOW.isoformat(), "exit_price": 11,
             "minute_bars": [bar(start, 10)]}
    store.append(prior)
    class FailingProvider:
        def fetch_session_close(self, *_):
            raise MassiveOvernightError("Temporary provider failure")
    refreshed = load_jev_evaluation([record()], refresh=True, outcome_store=store, provider=FailingProvider(), trade_store=NoTrades(), now=NOW + timedelta(minutes=1))
    assert refreshed["counts"]["completed"] == 1
    reloaded = JevOutcomeStore(data_dir=tmp_path)
    observation = reloaded.latest()["one"]
    assert observation["exit_price"] == 11 and observation["minute_bars"] == prior["minute_bars"]
    assert observation["exit_price_observed_at"] == NOW.isoformat()
    assert observation["refresh_issue"] == "Dated provider prices unavailable"
    cached = load_jev_evaluation([record()], outcome_store=reloaded, trade_store=NoTrades(), now=NOW + timedelta(minutes=2))
    assert cached["rows"][0]["net_return"] == .09
    assert cached["rows"][0]["labels"]["downside_before_close"] is None
