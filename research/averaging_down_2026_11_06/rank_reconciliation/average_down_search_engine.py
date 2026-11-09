"""Event-driven OHLC replay for a finite, predeclared add/exit policy grid.

No price scanning in Python per bar: NumPy finds the next qualifying node,
then the engine processes the earliest order crossing on that segment.
Open nodes are gaps and fill at their observed prices. Inside-bar crossings
are hypothetical limit/stop fills. Two OHLC orderings expose sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SearchPolicy:
    trigger_loss: int
    adds: tuple[int, ...]
    target_pnl: int | None
    loss_cap: int | None

    @property
    def budget(self):
        return 10000 + sum(self.adds)

    @property
    def key(self):
        return f"at{self.trigger_loss}_adds{'-'.join(map(str, self.adds)) or 'none'}_target{self.target_pnl}_stop{self.loss_cap}"


@dataclass
class PricePath:
    prices: np.ndarray
    gaps: np.ndarray
    labels: list[str]
    add_until: int
    original: dict
    deadline: str
    order: str


def price_path(row, bars, order, deadline="original"):
    from rank1_stop_loss_study import local_time

    prices, gaps, labels = [], [], []
    for bar in bars:
        when = local_time(bar).isoformat()
        prices.extend(bar[c.lower()] for c in order)
        gaps.extend([True, False, False, False])
        labels.extend([when] * 4)
    add_until = len(prices)
    prices.append(row["close"])
    gaps.append(True)
    labels.append(row["date"] + " close")
    if deadline == "original" and row["heldOvernight"]:
        prices.append(row["exit"])
        gaps.append(True)
        labels.append(row["next_session"] + " open")
    return PricePath(np.array(prices), np.array(gaps), labels, add_until, row, deadline, order)


def first_crossing(prices, start, level, above=False, end=None):
    selected = prices[start:end]
    if not len(selected):
        return None
    mask = selected >= level if above else selected <= level
    found = int(mask.argmax())
    return start + found if mask[found] else None


def replay(path, policy, cost=0.0):
    prices, gaps, row = path.prices, path.gaps, path.original
    spent = 10000.0
    qty = spent / (row["entry"] * (1 + cost))
    shares_after_cost = 1 - cost
    levels = [row["entry"] * (1 - policy.trigger_loss / 10000) ** (i + 1) for i in range(len(policy.adds))]
    cursor = stage = 0
    worst = 0.0
    entries = []
    activated = False
    reason, fill, at = "scheduled_exit", float(prices[-1]), len(prices) - 1

    def pnl_at(price):
        return qty * price * shares_after_cost - spent

    while cursor < len(prices):
        events = []
        if policy.loss_cap is not None:
            stop = (spent - policy.loss_cap) / (qty * shares_after_cost)
            index = first_crossing(prices, cursor, stop)
            if index is not None:
                events.append((index, stop, "stop"))
        if activated and policy.target_pnl is not None:
            target = (spent + policy.target_pnl) / (qty * shares_after_cost)
            index = first_crossing(prices, cursor, target, above=True)
            if index is not None:
                events.append((index, target, "target"))
        if stage < len(levels):
            index = first_crossing(prices, cursor, levels[stage], end=path.add_until)
            if index is not None:
                events.append((index, levels[stage], "add"))
        if not events:
            worst = min(worst, pnl_at(float(prices[cursor:].min())))
            break
        index = min(event[0] for event in events)
        same_node = [event for event in events if event[0] == index]
        if gaps[index]:
            event = min(same_node, key=lambda e: {"stop": 0, "target": 1, "add": 2}[e[2]])
            price = float(prices[index])
        else:
            descending = prices[index] < prices[index - 1]
            event = min(same_node, key=lambda e: (-e[1] if descending else e[1], e[2] != "stop"))
            price = event[1]
        if index > cursor:
            worst = min(worst, pnl_at(float(prices[cursor:index].min())))
        worst = min(worst, pnl_at(price))
        if event[2] != "add":
            reason = event[2] + ("_gap" if gaps[index] else "")
            fill, at = price, index
            break
        activated = True
        while stage < len(levels) and price <= levels[stage]:
            # A gap can cross several resting add limits at once. Each fills
            # at the observed opening price, subject to the total loss cap.
            cash = policy.adds[stage]
            if cash:
                qty += cash / (price * (1 + cost))
                spent += cash
                entries.append({"price": price, "cash": cash, "at": path.labels[index]})
            stage += 1
            worst = min(worst, pnl_at(price))
            if policy.loss_cap is not None and pnl_at(price) <= -policy.loss_cap:
                reason, fill, at = "stop_after_add_costs", price, index
                break
            if policy.target_pnl is not None and pnl_at(price) >= policy.target_pnl:
                reason, fill, at = "target_at_activation", price, index
                break
        if reason != "scheduled_exit":
            break
        # A gap node has been consumed; an intrabar threshold leaves the rest
        # of its segment available for another addition or a loss limit.
        cursor = index + 1 if gaps[index] else index
    pnl = round(pnl_at(fill), 2)
    if cost == 0 and reason == "scheduled_exit":
        baseline = row["pnl"] if path.deadline == "original" else round(10000 * (row["close"] / row["entry"] - 1), 2)
        pnl = baseline + sum(round(e["cash"] * (fill / e["price"] - 1), 2) for e in entries)
    if abs(pnl) < .005:
        pnl = 0.0
    return {"pnl": round(pnl, 2), "worst_mark": round(worst, 2), "capital": spent,
            "adds": len(entries), "activated": activated, "reason": reason,
            "exit_price": fill, "exit_at": path.labels[at], "entries": entries}


def policy_grid():
    triggers = [200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200]
    patterns = [(0,), (2500,), (5000,), (10000,), (20000,), (5000, 5000), (10000, 10000), (10000, 20000)]
    targets = [-500, -250, 0, 250, 500, 1000, 2000, None]
    caps = [500, 1000, 1500, 2000, 3000, 5000, None]
    policies = set()
    for trigger in triggers:
        for pattern in patterns:
            for target in targets:
                if target is not None and target <= -trigger:
                    continue
                for cap in caps:
                    if cap is not None and cap <= trigger:
                        continue
                    if pattern == (0,) and target is None:
                        continue
                    policies.add(SearchPolicy(trigger, pattern, target, cap))
    for cap in caps:
        policies.add(SearchPolicy(0, (), None, cap))
    return sorted(policies, key=lambda p: p.key)
