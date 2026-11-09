"""One equal-dollar add, a combined break-even exit, and optional total-loss cap.

Use the same 36 Rank 1 selections and opening-price assumptions as the repaired
dashboard. The add is a resting limit at a fixed decline from entry. A combined
break-even sell limit becomes active only after that trigger. No repeated adds
or re-entry. The existing close/next-open liquidation remains the deadline.

Minute OHLC cannot establish every order of intraminute events. Run both O-H-L-C
and O-L-H-C paths, report the P&L range, and use the worse of the two per trade
for the conservative table. These are execution sensitivities, not tick fills.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

from rank1_stop_loss_study import CAPITAL, closed_drawdown, local_time, regular_bars

ROOT = Path(__file__).resolve().parent


def refine_bars(ticker, bars):
    """Substitute cached second bars only when they reproduce the minute OHLC."""
    result, refined = [], []
    for bar in bars:
        path = ROOT / "second_bars" / f"{ticker}_{bar['t']}.json.gz"
        if not path.exists():
            result.append(bar)
            continue
        with gzip.open(path, "rt") as handle:
            doc = json.load(handle)
        seconds = sorted(doc["bars"], key=lambda b: b["t"])
        if not seconds:
            result.append(bar)
            continue
        assert doc["ticker"] == ticker and doc["adjusted"] is False
        assert all(bar["t"] <= second["t"] < bar["t"] + 60000 for second in seconds)
        for actual, expected in [(seconds[0]["o"], bar["o"]), (seconds[-1]["c"], bar["c"]),
                                 (min(b["l"] for b in seconds), bar["l"]), (max(b["h"] for b in seconds), bar["h"])]:
            assert abs(actual - expected) < 0.00001, (ticker, bar["t"], actual, expected)
        result.extend(seconds)
        refined.append({"ticker": ticker, "minute_et": local_time(bar).isoformat(), "seconds": len(seconds),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result, refined


@dataclass(frozen=True)
class Policy:
    trigger_loss: int
    add_dollars: int = CAPITAL
    loss_cap: int | None = None
    break_even_exit: bool = True

    @property
    def key(self):
        return f"trigger{self.trigger_loss}_add{self.add_dollars}_cap{self.loss_cap or 'none'}_be{int(self.break_even_exit)}"


def replay(row, bars, policy, cost=0.0, path_order="OHLC"):
    if policy.loss_cap is not None and policy.loss_cap <= policy.trigger_loss:
        raise ValueError("A total-loss cap must leave room beyond the loss that triggers the add")
    entry = row["entry"]
    trigger = entry * (1 - policy.trigger_loss / CAPITAL)
    spent = float(CAPITAL)
    qty = CAPITAL / (entry * (1 + cost))
    active = added = False
    worst = 0.0
    exit_price = exit_reason = exit_at = None
    add_price = add_at = trigger_at = None

    def mark(price):
        return qty * price * (1 - cost) - spent

    def target():
        return spent / (qty * (1 - cost))

    def stop():
        return None if policy.loss_cap is None else (spent - policy.loss_cap) / (qty * (1 - cost))

    def observe(price):
        nonlocal worst
        worst = min(worst, mark(price))

    def finish(price, reason, when):
        nonlocal exit_price, exit_reason, exit_at
        observe(price)
        exit_price, exit_reason, exit_at = price, reason, when

    def activate(price, when):
        nonlocal active, added, spent, qty, add_price, add_at, trigger_at
        active = True
        trigger_at = when
        if policy.add_dollars:
            qty += policy.add_dollars / (price * (1 + cost))
            spent += policy.add_dollars
            added, add_price, add_at = True, price, when
        observe(price)
        if policy.loss_cap is not None and mark(price) <= -policy.loss_cap:
            finish(price, "loss_limit_after_costs", when)

    def arrive(price, when, allow_add=True):
        """A new bar/opening auction may gap over an existing order."""
        observe(price)
        if policy.loss_cap is not None and mark(price) <= -policy.loss_cap:
            finish(price, "loss_limit_gap", when)
        elif active and policy.break_even_exit and price >= target():
            finish(price, "break_even_gap", when)
        elif allow_add and not active and price <= trigger:
            activate(price, when)

    def traverse(start, end, when):
        """Process the first threshold on a monotone segment before later prices."""
        price = start
        while exit_price is None:
            candidates = []
            stop_price = stop()
            if end < price:
                if stop_price is not None and end <= stop_price < price:
                    candidates.append((stop_price, "stop"))
                if not active and end <= trigger < price:
                    candidates.append((trigger, "activate"))
                candidates.sort(key=lambda event: (-event[0], event[1] != "stop"))
            elif end > price and active and policy.break_even_exit and price < target() <= end:
                candidates.append((target(), "target"))
            if not candidates:
                observe(end)
                return
            price, action = candidates[0]
            observe(price)
            if action == "stop":
                finish(price, "loss_limit", when)
            elif action == "target":
                finish(price, "break_even", when)
            else:
                activate(price, when)

    for bar in bars:
        when = local_time(bar).isoformat()
        arrive(bar["o"], when)
        if exit_price is not None:
            break
        path = [bar[letter.lower()] for letter in path_order]
        for start, end in zip(path, path[1:]):
            traverse(start, end, when)
            if exit_price is not None:
                break
        if exit_price is not None:
            break

    # Official close can differ slightly from the final minute's last trade.
    if exit_price is None:
        arrive(row["close"], row["date"] + "T16:00:00 ET", allow_add=False)
    if exit_price is None:
        deadline = row["next_session"] + " open" if row["heldOvernight"] else row["date"] + " close"
        arrive(row["exit"], deadline, allow_add=False)
        if exit_price is None:
            finish(row["exit"], "scheduled_exit", deadline)

    pnl = round(mark(exit_price), 2)
    if cost == 0 and exit_reason == "scheduled_exit":
        # Preserve the dashboard's rounding of the original leg.
        pnl = row["pnl"] + (round(policy.add_dollars * (exit_price / add_price - 1), 2) if added else 0)
    if abs(pnl) < .005:
        pnl = 0.0
    return {"date": row["date"], "ticker": row["ticker"], "policy": policy.key,
            "cost_bps_per_side": cost * 10000, "path_order": path_order,
            "original_pnl": row["pnl"], "triggered": active, "trigger_at": trigger_at,
            "added": added, "add_price": add_price, "add_at": add_at,
            "capital_committed": spent, "shares": qty, "combined_break_even_price": target(),
            "exit_price": exit_price, "exit_reason": exit_reason, "exit_at": exit_at,
            "pnl": round(pnl, 2), "worst_mark": round(worst, 2)}


def self_check():
    # Equal dollars at $100 and $95 give a harmonic, not arithmetic, average.
    qty = 10000 / 100 + 10000 / 95
    assert abs(20000 / qty - 97.43589743589743) < 1e-10
    assert abs(qty * 95 - 20000 + 500) < 1e-10
    row = {"date": "2026-09-25", "ticker": "TEST", "entry": 100, "close": 90,
           "exit": 90, "heldOvernight": False, "next_session": "2026-09-28", "pnl": -1000}
    b = lambda t, o, h, l, c: {"t": 1790343000000 + t * 60000, "o": o, "h": h, "l": l, "c": c}
    bars = [b(0, 100, 101, 94, 94), b(1, 94, 98, 94, 98)]
    result = replay(row, bars, Policy(500))
    assert result["added"] and result["pnl"] == 0 and result["exit_reason"] == "break_even"
    # High before adding cannot be reused as a later recovery.
    fall = [b(0, 100, 101, 94, 94), b(1, 94, 94, 88, 90)]
    assert replay(row, fall, Policy(500, loss_cap=1500), path_order="OHLC")["pnl"] == -1500
    assert replay(row, fall, Policy(500, loss_cap=1500), path_order="OLHC")["pnl"] == 0
    gap = [b(0, 100, 100, 94, 94), b(1, 85, 86, 80, 82)]
    assert replay(row, gap, Policy(500, loss_cap=1000))["pnl"] < -1000
    cost_result = replay(row, bars, Policy(500), cost=.0011)
    assert cost_result["combined_break_even_price"] > 20000 / qty and cost_result["pnl"] == 0
    try:
        replay(row, bars, Policy(500, loss_cap=500))
        raise AssertionError("An add at the total-loss cap must be rejected")
    except ValueError:
        pass


def aggregate(rows):
    values = [row["pnl"] for row in rows]
    added = [row for row in rows if row["added"]]
    return {"positions": len(rows), "total_pnl": round(sum(values), 2),
            "wins": sum(p > 0 for p in values), "flats": sum(p == 0 for p in values), "losses": sum(p < 0 for p in values),
            "triggered": sum(row["triggered"] for row in rows), "adds": len(added),
            "break_even_exits": sum(row["exit_reason"].startswith("break_even") for row in rows),
            "loss_limit_exits": sum(row["exit_reason"].startswith("loss_limit") for row in rows),
            "added_not_recovered": sum(row["pnl"] < 0 for row in added),
            "worst_trade": min(values), "worst_mark": min(row["worst_mark"] for row in rows),
            "closed_trade_max_drawdown": closed_drawdown(values),
            "max_capital_per_position": max(row["capital_committed"] for row in rows),
            "average_capital_committed": round(sum(row["capital_committed"] for row in rows) / len(rows), 2)}


def main():
    self_check()
    source = ROOT / "rank1_significance_rows.json"
    positions = json.loads(source.read_text())
    actuals = json.loads((ROOT / "overnight_actuals_snapshot.json").read_text())
    bars_by_date, refinements = {}, []
    for row in positions:
        row["next_session"] = next(r["next_session"] for r in actuals[row["date"]]["rows"] if r["ticker"] == row["ticker"])
        files = list((ROOT / "minute_bars").glob(f"{row['ticker']}_{row['date']}_*.json.gz"))
        assert len(files) == 1
        with gzip.open(files[0], "rt") as handle:
            doc = json.load(handle)
        assert doc["adjusted"] is False
        refined, evidence = refine_bars(row["ticker"], regular_bars(doc["bars"], row["date"]))
        bars_by_date[row["date"]] = refined
        refinements.extend(evidence)
    policies = []
    for trigger in [500, 600, 700, 1000]:
        policies.extend([Policy(trigger), Policy(trigger, add_dollars=0), Policy(trigger, break_even_exit=False)])
        for cap in [1000, 1500]:
            if cap > trigger:
                policies.extend([Policy(trigger, loss_cap=cap), Policy(trigger, add_dollars=0, loss_cap=cap)])
    policies.extend([Policy(250, loss_cap=500), Policy(250, add_dollars=0, loss_cap=500)])
    summaries, all_rows, ambiguities = [], [], []
    for cost in [0, .0011, .0026]:
        for policy in policies:
            conservative, favorable = [], []
            for row in positions:
                variants = [replay(row, bars_by_date[row["date"]], policy, cost, path) for path in ["OHLC", "OLHC"]]
                low, high = sorted(variants, key=lambda r: (r["pnl"], r["worst_mark"]))
                conservative.append(low)
                favorable.append(high)
                all_rows.append(low)
                if low["pnl"] != high["pnl"]:
                    ambiguities.append({"policy": policy.key, "cost_bps_per_side": cost * 10000,
                                        "date": row["date"], "ticker": row["ticker"], "lower": low, "upper": high})
            summary = {"policy": policy.key, "trigger_loss": policy.trigger_loss, "add_dollars": policy.add_dollars,
                       "loss_cap": policy.loss_cap, "break_even_exit": policy.break_even_exit,
                       "cost_bps_per_side": cost * 10000, **aggregate(conservative),
                       "favorable_path_total_pnl": aggregate(favorable)["total_pnl"],
                       "ambiguous_positions": sum(a["pnl"] != b["pnl"] for a, b in zip(conservative, favorable))}
            summaries.append(summary)
    result = {"baseline_pnl": round(sum(row["pnl"] for row in positions), 2),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "summary": summaries, "rows": all_rows, "ambiguous": ambiguities, "second_refinements": refinements,
              "assumptions": [
                  "Same 36 Rank 1 selections, hypothetical opening entries and original close/next-open deadlines; no re-entry.",
                  "Equal-dollar doubling: $10K initial cash plus one $10K addition; fractional shares match dashboard conventions.",
                  "Trigger losses refer to the original $10K position before costs; total-loss caps include both purchases and modeled costs.",
                  "The combined break-even target covers both cash outlays and any modeled entry/exit costs.",
                  "Cost cases are assumptions, not measured spreads; there is no financing or tax model.",
                  "A total loss cap at or below the original add trigger is invalid: the add does not reset the accumulated loss.",
                  "Resting limits fill at their prices or a better bar open; stop gaps fill at the worse opening price. Available depth is unverified.",
                  "Regular-hours orders only; no new purchase at the next-open liquidation deadline.",
                  "Both monotone OHLC paths are replayed; path-sensitive results are presented as ranges rather than assuming the favorable ordering.",
                  "Cached second bars replace ambiguous trigger minutes only after confirming they reproduce the original minute OHLC; remaining within-bar order is still a modeling limitation.",
                  "The conservative total selects the lower result per trade from the two modeled paths, not a hindsight trading signal.",
                  "The model's original ranking edge and post-decline edge are not assumed; this is a retrospective, unvalidated comparison.",
                  "The cached opening-entry feasibility and thin-market liquidity limitations apply; these are not actual broker executions.",
              ]}
    (ROOT / "rank1_average_down_study.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GROSS RESULTS")
    for r in summaries:
        if r["cost_bps_per_side"] == 0:
            print(r["policy"], "pnl", r["total_pnl"], "to", r["favorable_path_total_pnl"],
                  "adds", r["adds"], "BE", r["break_even_exits"], "stops", r["loss_limit_exits"],
                  "worst", r["worst_trade"], "worst_mark", r["worst_mark"], "ambiguous", r["ambiguous_positions"])
    print("Ambiguous trade-policy-cost cases", len(ambiguities))


if __name__ == "__main__":
    main()
