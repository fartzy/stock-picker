"""Forward Jev forecasts scored on dated prices, including untraded candidates.

No LLM is called here. Read-only evaluation uses cached observations; an explicit
refresh fetches completed outcomes. Forecast versions, models, list kinds and
entry bases remain separate. Raw Rank/Fit scores are never probabilities.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from stock_picker.ingestion.massive_overnight import MassiveOvernightClient, MassiveOvernightError
from stock_picker.ingestion.session import cash_session_window
from stock_picker.storage.jev_outcome_store import JevOutcomeStore
from stock_picker.storage.trade_store import TradeStore

EASTERN = ZoneInfo("America/New_York")
MINUTE = timedelta(minutes=1)
CLOSE_SETTLE = timedelta(minutes=15)
RELIABILITY_BINS = 5
POLICY = (
    "First successful forecast per session, ticker, list, question version, model, entry basis and cost/downside assumptions. "
    "Reassessments stay in the ledger but do not increase primary sample size. "
    "Baselines use strictly earlier completed sessions. Returns are hypothetical at the saved reference price, "
    "after the entered cost assumption. Morning questions are forecast at their recorded time, not available-at-open trade signals. "
    "No automatic buy, skip or size rule uses Jev."
)


def _time(value: str | None) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(value or "")
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (TypeError, ValueError):
        return None


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _probability(value: object) -> float | None:
    result = _number(value)
    return result if result is not None and 0 <= result <= 1 else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def downside_label(bars: list[dict], start: datetime, end: datetime, barrier: float) -> dict:
    """Whole post-decision minutes only; missing minutes cannot prove no touch.

    A touch in any observed whole minute establishes the event. A missing or
    straddling decision minute otherwise leaves the event unresolved.
    """
    first = start.replace(second=0, microsecond=0)
    if first < start:
        first += MINUTE
    expected = {int((first + index * MINUTE).timestamp() * 1000)
                for index in range(max(0, int((end - first).total_seconds() // 60)))}
    observed = {}
    partial_low = None
    partial_stamp = int(start.replace(second=0, microsecond=0).timestamp() * 1000)
    for bar in bars:
        stamp = _number(bar.get("t"))
        opened, high, low, closed = (_number(bar.get(key)) for key in ("o", "h", "l", "c"))
        if (stamp is None or stamp % 60_000 != 0 or any(value is None or value <= 0 for value in (opened, high, low, closed))
                or not low <= min(opened, closed) <= max(opened, closed) <= high):
            continue
        if stamp == partial_stamp:
            partial_low = low if partial_low is None else min(partial_low, low)
        if int(stamp) in expected:
            observed[int(stamp)] = min(observed.get(int(stamp), low), low)
    touched = any(low <= barrier for low in observed.values())
    partial_clear = first == start or (partial_low is not None and partial_low > barrier)
    complete = bool(expected) and set(observed) == expected and partial_clear
    return {
        "value": True if touched else False if complete else None,
        "observed_minutes": len(observed), "expected_minutes": len(expected),
        "partial_decision_minute": first != start,
        "reason": None if touched or complete else "Incomplete post-decision minute coverage",
    }


def _cohort(record: dict) -> tuple:
    target = record.get("target") or {}
    return (record.get("question_set_version"), record.get("model") or "unknown",
            record.get("kind"), target.get("entry_basis"),
            target.get("round_trip_cost_per_share"), target.get("downside_fraction"))


def _news_group(record: dict) -> str:
    state = record.get("state") or {}
    check = state.get("morning_news_judge") or state.get("news_evidence") or {}
    judge = state.get("news_judge") or {}
    if check.get("flag") or judge.get("flag"):
        return "flagged"
    if check.get("status") not in ("complete", "no_news"):
        return "not_checked"
    return "clear"


def _next_open(record: dict) -> datetime | None:
    day = (record.get("target") or {}).get("next_session")
    try:
        window = cash_session_window(datetime.combine(date.fromisoformat(day), datetime.min.time(), EASTERN))
        return window.open_at if window else None
    except (ValueError, TypeError):
        return None


def _base_row(record: dict, traded: set[tuple[str, str]]) -> dict:
    return {
        "forecast_id": record["id"], "ticker": record["ticker"], "session": record["session"],
        "kind": record.get("kind"), "question_set_version": record.get("question_set_version"),
        "model": record.get("model"), "entry_basis": (record.get("target") or {}).get("entry_basis"),
        "requested_at": record.get("requested_at"), "primary": False,
        "completed_at": record.get("completed_at"),
        "traded": (record["session"], record["ticker"]) in traded,
        "list_rank": record.get("list_rank"), "score": record.get("score"),
        "news_group": _news_group(record),
        "probabilities": {key: _probability(value) for key, value in (record.get("answers") or {}).items()},
        "status": "pending", "reason": "Outcome refresh pending", "net_return": None, "labels": {},
    }


def forecast_outcome(record: dict, observed: dict | None, now: datetime) -> dict:
    """Label a saved decision using supplied observations; never look up a trade."""
    target = record.get("target") or {}
    end, completed = _time(target.get("exit_at")), _time(record.get("completed_at"))
    version = record.get("question_set_version")
    if version == "morning_v1" and (not record.get("state") or not record.get("requested_at")):
        return {"status": "legacy", "reason": "Original morning decision snapshot unavailable",
                "net_return": None, "labels": {}}
    if record.get("issue"):
        return {"status": "unavailable", "reason": record["issue"], "net_return": None, "labels": {}}
    if end is None or completed is None or completed >= end or not record.get("state"):
        return {"status": "unavailable", "reason": "Missing or invalid forward decision snapshot",
                "net_return": None, "labels": {}}
    if now < end + CLOSE_SETTLE:
        return {"status": "pending", "reason": "Waiting for the target session to finish",
                "net_return": None, "labels": {}}
    if not observed:
        return {"status": "pending", "reason": "Refresh completed outcomes to fetch dated prices",
                "net_return": None, "labels": {}}
    if observed.get("issue"):
        return {"status": "unavailable", "reason": observed["issue"], "net_return": None, "labels": {}}
    price = _number(observed.get("close_price")) if version == "overnight_v1" else _number(target.get("entry_price"))
    exit_price = _number(observed.get("exit_price"))
    cost = _number(target.get("round_trip_cost_per_share"))
    if not price or price <= 0 or not exit_price or exit_price <= 0 or cost is None or cost < 0:
        return {"status": "unavailable", "reason": "Valid entry, exit and cost assumption required",
                "net_return": None, "labels": {}}
    net_return = (exit_price - price - cost) / price
    question = "net_positive_overnight" if version == "overnight_v1" else "buy_today" if version == "morning_v1" else "net_positive_by_close"
    labels = {question: net_return > 0}
    target_returns = {question: net_return}
    available_at = {question: (end + CLOSE_SETTLE).isoformat()}
    if version == "morning_v1":
        next_open = _number(observed.get("next_open"))
        next_at = _next_open(record)
        labels["hold_overnight"] = (next_open > exit_price if next_open is not None and next_at is not None
                                    and now >= next_at + CLOSE_SETTLE else None)
        target_returns["hold_overnight"] = (next_open / exit_price - 1) if labels["hold_overnight"] is not None else None
        available_at["hold_overnight"] = (next_at + CLOSE_SETTLE).isoformat() if next_at else None
    detail = {}
    if record["question_set_version"] == "entry_v1":
        start = _time(target.get("barrier_start_at")) or completed
        barrier = _number(target.get("downside_price"))
        if barrier is not None and barrier > 0 and completed <= start < end:
            detail = downside_label(observed.get("minute_bars") or [], start, end, barrier)
            labels["downside_before_close"] = detail["value"]
        else:
            labels["downside_before_close"] = None
        available_at["downside_before_close"] = (end + CLOSE_SETTLE).isoformat()
        target_returns["downside_before_close"] = net_return
    return {"status": "completed", "reason": None, "net_return": net_return,
            "labels": labels, "target_returns": target_returns, "label_available_at": available_at,
            "downside_coverage": detail, "observed_at": observed.get("observed_at"),
            "refresh_issue": observed.get("refresh_issue")}


def _target_summary(question: str, rows: list[dict]) -> dict:
    valid = [row for row in rows if row["labels"].get(question) is not None
             and row["probabilities"].get(question) is not None]
    probabilities = [row["probabilities"][question] for row in valid]
    outcomes = [float(row["labels"][question]) for row in valid]
    reliability = []
    for index in range(RELIABILITY_BINS):
        lower, upper = index / RELIABILITY_BINS, (index + 1) / RELIABILITY_BINS
        members = [row for row in valid if lower <= row["probabilities"][question]
                   and (row["probabilities"][question] < upper or index == RELIABILITY_BINS - 1)]
        returns = [value for row in members if (value := row.get("target_returns", {}).get(question)) is not None]
        reliability.append({"lower": lower, "upper": upper, "n": len(members),
                            "mean_probability": _mean([row["probabilities"][question] for row in members]),
                            "event_rate": _mean([float(row["labels"][question]) for row in members]),
                            "mean_net_return": _mean(returns),
                            "loss_rate": _mean([float(value < 0) for value in returns])})
    # A day never supplies labels to its own baseline, even when hundreds of names share it.
    baseline_errors, matched_jev_errors = [], []
    for row in valid:
        decision = _time(row.get("completed_at"))
        prior_labels = [float(prior["labels"][question]) for prior in valid if prior["session"] < row["session"]
                        and decision is not None and (available := _time(prior.get("label_available_at", {}).get(question))) is not None
                        and available <= decision]
        if prior_labels:
            baseline = statistics.fmean(prior_labels)
            baseline_errors.append((baseline - float(row["labels"][question])) ** 2)
            matched_jev_errors.append((row["probabilities"][question] - float(row["labels"][question])) ** 2)
    return {"question": question, "n": len(valid), "brier": _mean([(p - y) ** 2 for p, y in zip(probabilities, outcomes)]),
            "mean_probability": _mean(probabilities), "event_rate": _mean(outcomes), "reliability": reliability,
            "prior_day_baseline_brier": _mean(baseline_errors), "matched_jev_brier": _mean(matched_jev_errors),
            "baseline_n": len(baseline_errors), "unresolved": len(rows) - len(valid)}


def _returns(rows: list[dict]) -> dict:
    values = [row["net_return"] for row in rows if row["net_return"] is not None]
    return {"n": len(values), "mean_net_return": _mean(values),
            "win_rate": _mean([float(value > 0) for value in values]),
            "losses": sum(value < 0 for value in values), "worst_net_return": min(values) if values else None}


def summarize_jev(records: list[dict], observations: dict[str, dict], *, now: datetime,
                  traded: set[tuple[str, str]] | None = None) -> dict:
    rows, seen = [], set()
    duplicate_count = 0
    grouped = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.get("completed_at") or "", item["id"])):
        row = _base_row(record, traded or set())
        row.update(forecast_outcome(record, observations.get(record["id"]), now))
        key = (record["session"], record["ticker"], *_cohort(record))
        successful = not record.get("issue") and any(p is not None for p in row["probabilities"].values())
        row["primary"] = successful and key not in seen and row["status"] != "legacy"
        duplicate_count += successful and key in seen and row["status"] != "legacy"
        if row["primary"]:
            seen.add(key)
            grouped[_cohort(record)].append(row)
        rows.append(row)
    cohorts = []
    for (version, model, kind, basis, costs, downside), members in sorted(grouped.items(), key=lambda item: str(item[0])):
        comparisons = []
        for name, subset in (("All candidates", members), ("Traded", [r for r in members if r["traded"]]),
                             ("Untraded", [r for r in members if not r["traded"]]),
                             ("Morning news flagged", [r for r in members if r["news_group"] == "flagged"]),
                             ("Morning news clear", [r for r in members if r["news_group"] == "clear"]),
                             ("Morning news not fully checked", [r for r in members if r["news_group"] == "not_checked"]),
                             ("List rank 1 to 5", [r for r in members if r.get("list_rank") and r["list_rank"] <= 5])):
            comparisons.append({"name": name, **_returns(subset)})
        questions = sorted({q for row in members for q in row["probabilities"]
                            if q in ("net_positive_by_close", "downside_before_close", "net_positive_overnight", "buy_today", "hold_overnight")})
        cohorts.append({"question_set_version": version, "model": model, "kind": kind, "entry_basis": basis,
                        "records": len(members), "completed": sum(r["status"] == "completed" for r in members),
                        "round_trip_cost_per_share": costs, "downside_fraction": downside,
                        "sessions": len({r["session"] for r in members if r["status"] == "completed"}),
                        "returns": _returns(members), "targets": [_target_summary(q, members) for q in questions],
                        "comparisons": comparisons})
    primary = [row for row in rows if row["primary"]]
    return {"generated_at": now.isoformat(), "forecast_count": len(rows), "primary_count": len(primary),
            "duplicate_count": duplicate_count,
            "legacy_count": sum(r["status"] == "legacy" for r in rows),
            "failed_count": sum(bool(record.get("issue")) for record in records),
            "counts": {name: sum(r["status"] == name for r in primary) for name in ("completed", "pending", "unavailable")},
            "cohorts": cohorts, "rows": rows, "policy": POLICY}


def _overnight_prices(provider: MassiveOvernightClient, ticker: str, session: date,
                      next_day: date, now: datetime, cache: dict) -> tuple[float, float]:
    key = ("overnight", ticker, session, next_day)
    if key not in cache:
        if next_day == now.astimezone(EASTERN).date():
            closed = provider.fetch_session_close(ticker, session)
            opened = provider.fetch_current_open(ticker, next_day)
            if closed.session != session or not math.isclose(closed.price, opened.previous_close, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("Verified close and snapshot prior close disagree")
            cache[key] = (closed.price, opened.open, opened.corporate_action)
        else:
            verified = provider.fetch(ticker, session, next_day)
            matches = {stamp.date(): (stamp, row) for stamp, row in verified.history.iterrows()}
            if session not in matches or next_day not in matches:
                raise ValueError("Dated close or next-session open missing")
            stamp, row = matches[next_day]
            cache[key] = (float(matches[session][1]["Close"]), float(row["Open"]),
                          verified.provenance.loc[stamp, "corporate_action"])
    closed, opened, action = cache[key]
    if action != "verified_none":
        raise ValueError("Corporate action prevents a simple overnight price comparison")
    return closed, opened


def _observe(record: dict, provider: MassiveOvernightClient, now: datetime, cache: dict) -> dict:
    target = record["target"]
    ticker, session = record["ticker"], date.fromisoformat(record["session"])
    observation = {"forecast_id": record["id"], "observed_at": now.isoformat(), "source": "massive_unadjusted"}
    try:
        if record["question_set_version"] == "overnight_v1":
            next_day = date.fromisoformat(target["next_session"])
            closed, opened = _overnight_prices(provider, ticker, session, next_day, now, cache)
            observation.update(close_price=closed, exit_price=opened)
        else:
            key = ("close", ticker, session)
            if key not in cache:
                close = provider.fetch_session_close(ticker, session)
                if close.session != session:
                    raise ValueError("Close session mismatch")
                cache[key] = close.price
            observation["exit_price"] = cache[key]
            if record["question_set_version"] == "morning_v1":
                next_at = _next_open(record)
                if next_at is not None and now >= next_at + CLOSE_SETTLE:
                    try:
                        closed, opened = _overnight_prices(provider, ticker, session, date.fromisoformat(target["next_session"]), now, cache)
                        if math.isclose(closed, observation["exit_price"], rel_tol=1e-4, abs_tol=1e-4):
                            observation["next_open"] = opened
                    except (MassiveOvernightError, ValueError, KeyError, TypeError):
                        observation["overnight_issue"] = "Verified next-open comparison unavailable"
                return observation
            key = ("minutes", ticker, session)
            if key not in cache:
                try:
                    cache[key] = (provider.fetch_minute_bars(ticker, session), None)
                except MassiveOvernightError:
                    cache[key] = ([], "Post-decision minute prices unavailable")
            observation["minute_bars"], observation["minute_issue"] = cache[key]
    except MassiveOvernightError:
        observation["issue"] = "Dated provider prices unavailable"
    except (ValueError, KeyError, TypeError):
        observation["issue"] = "Outcome price or corporate-action validation failed"
    return observation


def _merge_observation(previous: dict | None, current: dict) -> dict:
    """A failed refresh is an observation of availability, not a lost price."""
    if previous is None:
        return current
    merged = {**previous, **current}
    if current.get("issue") and previous.get("exit_price") is not None:
        merged["refresh_issue"] = current["issue"]
        merged.pop("issue", None)
    elif not current.get("issue"):
        merged.pop("issue", None)
        merged.pop("refresh_issue", None)
    if current.get("minute_issue"):
        merged["refresh_issue"] = current["minute_issue"]
    for field in ("exit_price", "close_price", "next_open"):
        if current.get(field) is not None:
            merged[f"{field}_observed_at"] = current["observed_at"]
        elif previous.get(field) is not None:
            merged[field] = previous[field]
            merged[f"{field}_observed_at"] = previous.get(f"{field}_observed_at", previous["observed_at"])
    prior_bars = previous.get("minute_bars") or []
    current_bars = current.get("minute_bars") or []
    if prior_bars:
        # Keep conflicting revisions for the conservative path labeler, which
        # retains any valid observed touch rather than erasing an earlier low.
        merged["minute_bars"] = prior_bars + [bar for bar in current_bars if bar not in prior_bars]
    return merged


def _fully_resolved(record: dict, observation: dict | None, now: datetime) -> bool:
    outcome = forecast_outcome(record, observation, now)
    required = {"entry_v1": ("net_positive_by_close", "downside_before_close"),
                "overnight_v1": ("net_positive_overnight",),
                "morning_v1": ("buy_today", "hold_overnight")}.get(record.get("question_set_version"), ())
    return bool(required) and all(outcome["labels"].get(question) is not None for question in required)


def load_jev_evaluation(records: list[dict], *, refresh: bool = False,
                        outcome_store: JevOutcomeStore | None = None,
                        provider: MassiveOvernightClient | None = None,
                        trade_store: TradeStore | None = None,
                        now: datetime | None = None) -> dict:
    clock = now or datetime.now(timezone.utc)
    store = outcome_store if outcome_store is not None else JevOutcomeStore()
    observations = store.latest()
    if refresh:
        client = provider if provider is not None else MassiveOvernightClient()
        cache = {}
        for record in records:
            end = _time((record.get("target") or {}).get("exit_at"))
            if record.get("question_set_version") not in ("entry_v1", "overnight_v1", "morning_v1") or record.get("issue") or end is None:
                continue
            window = cash_session_window(end)
            if window is None or clock < end + CLOSE_SETTLE:
                continue
            if _fully_resolved(record, observations.get(record["id"]), clock):
                continue
            observation = _merge_observation(observations.get(record["id"]), _observe(record, client, clock, cache))
            store.append(observation)
            observations[record["id"]] = observation
    trades = (trade_store if trade_store is not None else TradeStore()).read()
    traded = {(stamp.astimezone(EASTERN).date().isoformat(), str(row["ticker"]))
              for row in trades.to_dict(orient="records") if (stamp := _time(row.get("executed_at"))) is not None}
    return summarize_jev(records, observations, now=clock, traded=traded)
