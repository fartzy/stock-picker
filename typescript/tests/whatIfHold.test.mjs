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

test("a zero gain sells at the next morning open when it is at or above the close", () => {
  const exit = holdExit(4.0, achr, 0, 10, 1);
  assert.equal(exit.reason, "target");
  assert.ok(Math.abs(exit.returnPct - (4.01 / 4.0 - 1)) < 1e-12);
});

test("an open above the target fills at the open even if the low later hits the stop", () => {
  const exit = holdExit(10, [{ date: "2026-10-15", open: 10.2, high: 10.3, low: 8.5, close: 8.8 }], 0, 10, 5);
  assert.equal(exit.reason, "target");
  assert.ok(Math.abs(exit.returnPct - 0.02) < 1e-12);
});

test("a gap down through the stop sells at the open, not the stop", () => {
  const exit = holdExit(10, [{ date: "2026-10-15", open: 8, high: 8.2, low: 7.5, close: 8.1 }], 3, 10, 1);
  assert.equal(exit.reason, "stop");
  assert.ok(Math.abs(exit.returnPct + 0.2) < 1e-12);
});

test("a drop and gain in the same session sell at the stop", () => {
  const exit = holdExit(10, [{ date: "2026-10-15", open: 10, high: 11, low: 8.5, close: 10.5 }], 3, 10, 1);
  assert.equal(exit.reason, "stop");
  assert.ok(Math.abs(exit.returnPct + 0.1) < 1e-12);
});

test("neither level sells at the deadline close", () => {
  const exit = holdExit(4.07, achr, 6, 10, 2);
  assert.equal(exit.reason, "deadline");
  assert.equal(exit.day, 2);
  assert.ok(Math.abs(exit.returnPct) < 1e-12);
});

const held = (extra) => ({
  ticker: "ACHR", open_price: 4.32, close_price: 4.07,
  hypothetical: { status: "scored", endingValue: 9421.3 }, ...extra,
});

test("hold totals keep the day trade and add the after-close move on the same shares", () => {
  const total = summarizeHold([held({})], new Map([["ACHR", achr]]), 3, 10, 1, 10000);
  const after = 10000 / 4.32 * 4.07 * 0.03;
  assert.equal(total.n, 1);
  assert.ok(Math.abs(total.pnl - (9421.3 + after - 10000)) < 0.01);
});

test("hold totals use the same scored and held picks as the close-only baseline", () => {
  const total = summarizeHold([
    held({ hypothetical: { status: "skipped", endingValue: 9000 } }),
    held({ hypothetical: { status: "pending", endingValue: null } }),
  ], new Map([["ACHR", achr]]), 3, 10, 1, 10000);
  assert.equal(total, null);
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

test("a wait longer than the saved bars stays pending until a level prints", () => {
  assert.equal(holdExit(4.07, achr, 6, 10, 3), null);
  assert.equal(holdExit(4.07, achr, 1, 10, 3).day, 1);
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
