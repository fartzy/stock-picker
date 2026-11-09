"""Replay fixed dollar stops on the 36 repaired Rank 1 dashboard positions.

Keep the original $10K opening entry and close/next-open exit rule. A stop is
active during the entry day's regular session. A triggered stop exits all
shares, with no re-entry. With no intraday trigger, retain the dashboard exit;
in particular, an overnight gap is sold at the next opening price, not at an
unavailable stop price. These are hypothetical executions, not broker fills.

Two fill models are reported: exact trigger fills, and minute-open gaps filled
at the worse bar open. Neither model measures spreads, intraminute jumps,
available depth, financing, commissions, or market impact.
"""
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
EASTERN = ZoneInfo("America/New_York")
CAPITAL = 10_000
LIMITS = (500, 600, 700, 1000)


def local_time(bar):
    return datetime.fromtimestamp(bar["t"] / 1000, EASTERN)


def regular_bars(bars, session):
    return sorted([
        bar for bar in bars
        if local_time(bar).date().isoformat() == session
        and 570 <= local_time(bar).hour * 60 + local_time(bar).minute < 960
    ], key=lambda bar: bar["t"])


def first_stop(bars, stop_price):
    for bar in bars:
        if bar["l"] <= stop_price:
            return {"time_et": local_time(bar).isoformat(), "price": min(stop_price, bar["o"]),
                    "bar_open": bar["o"], "bar_low": bar["l"], "bar_volume": bar.get("v"),
                    "minute_open_gap": bar["o"] < stop_price}
    return None


def closed_drawdown(values):
    balance = peak = worst = 0
    for pnl in values:
        balance += pnl
        peak = max(peak, balance)
        worst = max(worst, peak - balance)
    return round(worst, 2)


def summary(values):
    total = round(sum(values), 2)
    return {"positions": len(values), "total_pnl": total, "period_gain_pct": total / CAPITAL * 100,
            "mean_pnl": round(mean(values), 2), "wins": sum(value > 0 for value in values),
            "losses": sum(value < 0 for value in values), "worst_trade": min(values),
            "standard_deviation": round(stdev(values), 2),
            "closed_trade_max_drawdown": closed_drawdown(values)}


def validate_simulation():
    bars = [{"t": 0, "o": 100, "l": 96}, {"t": 60000, "o": 96, "l": 94},
            {"t": 120000, "o": 90, "l": 85}]
    assert first_stop(bars, 95)["price"] == 95  # First touch, no future-bar hindsight.
    assert first_stop(bars, 93)["price"] == 90  # A gap cannot fill at the trigger.
    assert first_stop(bars, 80) is None
    assert closed_drawdown([100, -500, 200, -100]) == 500


def main():
    validate_simulation()
    source = ROOT / "rank1_significance_rows.json"
    positions = json.loads(source.read_text())
    assert len(positions) == 36 and round(sum(row["pnl"] for row in positions), 2) == 2708.87
    daily = {(row["date"], row["ticker"]): row["payload"]
             for row in json.loads((ROOT / "daily_price_snapshot.json").read_text())}
    records, coverage = [], []
    for row in positions:
        paths = list((ROOT / "minute_bars").glob(f"{row['ticker']}_{row['date']}_*.json.gz"))
        assert len(paths) == 1, (row["date"], row["ticker"], paths)
        with gzip.open(paths[0], "rt") as handle:
            document = json.load(handle)
        assert document["adjusted"] is False
        assert document["ticker"] == row["ticker"] and document["entry_date"] == row["date"]
        bars = regular_bars(document["bars"], row["date"])
        assert bars and len({bar["t"] for bar in bars}) == len(bars)
        assert all(0 < bar["l"] <= min(bar["o"], bar["c"]) <= max(bar["o"], bar["c"]) <= bar["h"] for bar in bars)
        low = min(bar["l"] for bar in bars)
        raw_daily = daily[(row["date"], row["ticker"])]
        assert abs(low - raw_daily["low"]) < 0.00001
        assert abs(row["entry"] / raw_daily["open"] - 1) < 0.0001
        assert abs(row["close"] / raw_daily["close"] - 1) < 0.0001
        coverage.append({"date": row["date"], "ticker": row["ticker"], "bars": len(bars),
                         "first_bar_et": local_time(bars[0]).isoformat(),
                         "last_bar_et": local_time(bars[-1]).isoformat(),
                         "minute_low_matches_daily_low": True,
                         "minute_file": str(paths[0].relative_to(ROOT)),
                         "minute_file_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest()})
        for limit in LIMITS:
            trigger_price = row["entry"] * (1 - limit / CAPITAL)
            hit = first_stop(bars, trigger_price)
            pnl = round(CAPITAL * (hit["price"] / row["entry"] - 1), 2) if hit else row["pnl"]
            records.append({"date": row["date"], "ticker": row["ticker"], "loss_limit": limit,
                            "entry_price": row["entry"], "original_exit_price": row["exit"],
                            "original_pnl": row["pnl"], "original_held_overnight": row["heldOvernight"],
                            "lowest_intraday_price": low,
                            "maximum_intraday_loss": round(max(0, CAPITAL * (1 - low / row["entry"])), 2),
                            "stop_price": trigger_price, "stop_triggered_intraday": hit is not None,
                            "stop_fill": hit, "modeled_pnl": pnl,
                            "ideal_trigger_pnl": -limit if hit else row["pnl"],
                            "change_vs_original": round(pnl - row["pnl"], 2),
                            "winner_stopped": bool(hit and row["pnl"] > 0),
                            "overnight_exit_beyond_limit": bool(not hit and row["heldOvernight"] and row["pnl"] < -limit),
                            "all_in_costs_included": False})
    comparisons = []
    for limit in LIMITS:
        selected = [row for row in records if row["loss_limit"] == limit]
        improves = [row for row in selected if row["change_vs_original"] > 0]
        worsens = [row for row in selected if row["change_vs_original"] < 0]
        gains = round(sum(row["change_vs_original"] for row in improves), 2)
        harms = round(-sum(row["change_vs_original"] for row in worsens), 2)
        aggregate = summary([row["modeled_pnl"] for row in selected])
        assert round(2708.87 + gains - harms, 2) == aggregate["total_pnl"]
        comparisons.append({"loss_limit": limit, "entry_price_drop_pct": limit / CAPITAL * 100,
                            **aggregate,
                            "ideal_trigger_total_pnl": round(sum(row["ideal_trigger_pnl"] for row in selected), 2),
                            "stop_triggers": sum(row["stop_triggered_intraday"] for row in selected),
                            "winner_stops": sum(row["winner_stopped"] for row in selected),
                            "improved_trades": len(improves), "worsened_trades": len(worsens),
                            "saved_on_improved_trades": gains, "lost_on_worsened_trades": harms,
                            "change_vs_original": round(aggregate["total_pnl"] - 2708.87, 2),
                            "minute_open_gaps": [row for row in selected if row["stop_fill"] and row["stop_fill"]["minute_open_gap"]],
                            "overnight_gaps_past_limit": [row for row in selected if row["overnight_exit_beyond_limit"]],
                            "winners_stopped": [row for row in selected if row["winner_stopped"]]})
    result = {
        "scenario": "Rank 1, $10,000 per position; same-day close for non-losers, next open for losers; Sep 18–Nov 6, 2026",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline": summary([row["pnl"] for row in positions]), "comparisons": comparisons,
        "coverage": coverage, "rows": records,
        "assumptions": [
            "The position is bought at the repaired dashboard opening price, using fractional shares as the dashboard does.",
            "The loss limit is fixed relative to the initial purchase cost, not a trailing stop from the highest price.",
            "Stops are monitored during the entry session's regular hours only, including the first available opening minute.",
            "Sell the entire position at the first qualifying minute low crossing; no re-entry or averaging down.",
            "Fill at the trigger unless the minute opens below it, in which case fill at that lower opening price.",
            "Unstopped positions follow the original close/next-open exit, including overnight gaps through the loss limit.",
            "Minute bars locate a stop touch within one minute; they do not prove the executable bid or exact intraminute path.",
            "No spread, commission, slippage beyond observed minute-open gaps, financing, or tax is deducted.",
            "These thresholds were examined after seeing these dates; this is an in-sample what-if, not an out-of-sample forecast.",
            "Opening entries may precede pick publication; the result measures the displayed scenario, not verified achievable live fills.",
            "No bar is invented for sparse minutes: all 36 session lows match the separate raw daily feed; sparse trading still limits fill certainty.",
        ],
    }
    (ROOT / "rank1_stop_loss_study.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baseline": result["baseline"], "comparisons": [
        {key: value for key, value in comparison.items() if key not in {"minute_open_gaps", "overnight_gaps_past_limit", "winners_stopped"}}
        for comparison in comparisons]}, indent=2))
    print("STOP DETAILS")
    for row in records:
        if row["stop_triggered_intraday"] or row["overnight_exit_beyond_limit"]:
            print(row["loss_limit"], row["date"], row["ticker"], row["original_pnl"], row["modeled_pnl"],
                  row["stop_fill"]["time_et"] if row["stop_fill"] else "overnight open", "MAE", row["maximum_intraday_loss"])


if __name__ == "__main__":
    main()
