import assert from "node:assert/strict";
import test from "node:test";

import { addHoldSummaries, holdExit, resolveAfterClose, summarizeHold } from "../src/whatIfHold.js";

const achr = [
  { date: "2026-10-15", open: 4.01, high: 4.29, low: 3.97, close: 4.27 },
  { date: "2026-10-16", open: 4.30, high: 4.31, low: 4.01, close: 4.07 },
];

test("a gain target sells at the target even after a dip", () => {
  const exit = holdExit(4.07, achr, 3, 10, 1);
  assert.equal(exit.reason, "target");
  assert.equal(exit.day, 1);
  assert.ok(Math.abs(exit.returnPct - 0.03) < 1e-12);
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

const choice = (overrides) => ({ mode: "target", gainPct: 0, dropPct: 10, days: 1, rule: "down", portion: 30, ...overrides });

test("a zero gain one-day wait resolves to next morning open with every pick held", () => {
  assert.deepEqual(resolveAfterClose(choice({})), { nextOpen: { rule: "all", portion: 100 }, hold: null });
  assert.deepEqual(
    resolveAfterClose(choice({ mode: "next-open", rule: "all", portion: 100 })),
    resolveAfterClose(choice({})),
  );
});

test("other gains and waits keep the hold rule", () => {
  assert.deepEqual(resolveAfterClose(choice({ gainPct: 1 })), { nextOpen: null, hold: { gainPct: 1, dropPct: 10, days: 1 } });
  assert.deepEqual(resolveAfterClose(choice({ days: 2 })), { nextOpen: null, hold: { gainPct: 0, dropPct: 10, days: 2 } });
  assert.deepEqual(resolveAfterClose(choice({ mode: "close" })), { nextOpen: null, hold: null });
});
