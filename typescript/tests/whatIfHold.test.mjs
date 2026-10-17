import assert from "node:assert/strict";
import test from "node:test";

import { addHoldSummaries, holdExit } from "../src/whatIfHold.js";

const achr = [
  { date: "2026-10-15", high: 4.29, low: 3.97, close: 4.27 },
  { date: "2026-10-16", high: 4.31, low: 4.01, close: 4.07 },
];

test("a gain target sells at the target even after a dip", () => {
  const exit = holdExit(4.07, achr, 3, 10, 1);
  assert.equal(exit.reason, "target");
  assert.equal(exit.day, 1);
  assert.ok(Math.abs(exit.returnPct - 0.03) < 1e-12);
});

test("a drop that prints before the gain sells at the stop", () => {
  const exit = holdExit(10, [{ date: "2026-10-15", high: 11, low: 8.5, close: 10.5 }], 3, 10, 1);
  assert.equal(exit.reason, "stop");
  assert.ok(Math.abs(exit.returnPct + 0.1) < 1e-12);
});

test("neither level sells at the deadline close", () => {
  const exit = holdExit(4.07, achr, 6, 10, 2);
  assert.equal(exit.reason, "deadline");
  assert.equal(exit.day, 2);
  assert.ok(Math.abs(exit.returnPct) < 1e-12);
});

test("strategy totals add the priced days and keep the pending count", () => {
  const total = addHoldSummaries([
    { pnl: 100, avg: 0.01, n: 2, pending: 0 },
    { pnl: -50, avg: -0.01, n: 1, pending: 3 },
  ]);
  assert.equal(total.pnl, 50);
  assert.equal(total.n, 3);
  assert.equal(total.pending, 3);
  assert.ok(Math.abs(total.avg - (0.02 - 0.01) / 3) < 1e-12);
});

test("a wait longer than the saved bars stays pending", () => {
  assert.equal(holdExit(4.07, achr, 1, 10, 3), null);
});
