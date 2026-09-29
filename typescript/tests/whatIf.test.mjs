import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_TRADE_SIZE, simulateBook, simulatePick, simulationDay } from "../src/whatIf.js";

const pick = (session_return, extra = {}) => ({
  ticker: "AAA", rank: 1, predicted: 0.01, open_price: 100, close_price: null,
  session_return, ...extra,
});
const day = (as_of, rank = [], fit = [], extra = {}) => ({
  as_of, rank, fit, rank_avg: null, fit_avg: null, scan_id: "", ...extra,
});
const book = (days) => ({
  days, fit_compound: null, rank_compound: null, fit_days: 0, rank_days: 0,
  kind: "both", top_k: null, n_picks: 0,
});
const options = {
  kind: "both", fitTopK: 2, rankTopK: 2, dollarsPerTrade: DEFAULT_TRADE_SIZE,
  lookbackDays: null, asOf: "2026-09-29",
};

test("default is $10K per trade, including losses and flat trades", () => {
  assert.equal(DEFAULT_TRADE_SIZE, 10_000);
  assert.deepEqual(simulatePick(pick(0.02), 10_000), { status: "scored", pnl: 200, endingValue: 10_200 });
  assert.deepEqual(simulatePick(pick(-0.01), 10_000), { status: "scored", pnl: -100, endingValue: 9_900 });
  assert.deepEqual(simulatePick(pick(0), 5_000), { status: "scored", pnl: 0, endingValue: 5_000 });
});

test("skipped and pending picks never become zero-return investments", () => {
  for (const value of [null, NaN, Infinity]) {
    assert.equal(simulatePick(pick(value), 10_000).status, "pending");
  }
  assert.deepEqual(simulatePick(pick(0.5, { news_blocks: true }), 10_000), {
    status: "skipped", pnl: null, endingValue: null,
  });
  for (const amount of [0, -1, NaN, Infinity]) assert.throws(() => simulatePick(pick(0.01), amount), RangeError);
});

test("top 2 means $20K per strategy per day; strategies stay independent", () => {
  const data = book([day("2026-09-29", [pick(0.02), pick(-0.01, { rank: 2 }), pick(0.9, { rank: 3 })],
    [pick(0.03), pick(0.01, { rank: 2 })])]);
  const result = simulateBook(data, options);
  assert.deepEqual(result.rank_money, { pnl: 100, invested: 20_000, endingValue: 20_100, completed: 2, pending: 0 });
  assert.deepEqual(result.fit_money, { pnl: 400, invested: 20_000, endingValue: 20_400, completed: 2, pending: 0 });
  const half = simulateBook(data, { ...options, dollarsPerTrade: 5_000 });
  assert.equal(half.rank_money.pnl, 50);
  assert.equal(half.fit_money.pnl, 200);
  assert.equal(half.days[0].rank[0].hypothetical.endingValue, 5_100);
  assert.equal(data.days[0].rank[0].hypothetical, undefined); // Pure, no API-data mutation.
});

test("dollars are additive, not compound profits or an average times top K", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(-0.1)]),
    day("2026-09-28", [pick(0.1), pick(0.1, { rank: 2 })]),
  ]), options);
  assert.equal(result.rank_money.pnl, 1_000);
  assert.ok(Math.abs(result.rank_compound - (-0.01)) < 1e-12);
  assert.equal(result.rank_stats.n_scored, 3);
});

test("news blocks do not backfill below top K; pending closes do not dilute totals", () => {
  const result = simulateBook(book([day("2026-09-29", [
    pick(0.9, { news_blocks: true }), pick(null, { rank: 2 }), pick(0.1, { rank: 3 }),
  ])]), options);
  assert.equal(result.rank_money.pnl, null);
  assert.equal(result.rank_money.endingValue, null);
  assert.equal(result.rank_money.invested, 0);
  assert.equal(result.rank_money.pending, 1);
  assert.equal(result.rank_stats.n_avoid, 1);
  assert.equal(result.rank_days, 0);
});

test("line-item cents reconcile with day and period totals", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.0000014), pick(0.0000014, { rank: 2 })]),
    day("2026-09-28", [pick(-0.0000014)]),
  ]), options);
  assert.equal(result.days[0].rank_money.pnl, 0.02);
  assert.equal(result.days[1].rank_money.pnl, -0.01);
  assert.equal(result.rank_money.pnl, 0.01);
  assert.equal(result.rank_money.endingValue, 30_000.01);
});

test("rolling 2/3 week windows include today and the correct lower boundary", () => {
  const data = book(["2026-09-30", "2026-09-29", "2026-09-16", "2026-09-15", "2026-09-09", "2026-09-08"]
    .map((as_of) => day(as_of, [pick(0.01)])));
  const two = simulateBook(data, { ...options, lookbackDays: 14 });
  assert.deepEqual(two.days.map((row) => row.as_of), ["2026-09-29", "2026-09-16"]);
  assert.equal(two.from, "2026-09-16");
  assert.equal(two.rank_money.pnl, 200);
  const three = simulateBook(data, { ...options, lookbackDays: 21 });
  assert.equal(three.from, "2026-09-09");
  assert.equal(three.rank_money.pnl, 400);
  assert.equal(simulateBook(data, options).rank_money.pnl, 500); // Future day never included.
});

test("date windows cross month/year boundaries and use Chicago's day", () => {
  assert.equal(simulationDay(new Date("2026-09-30T02:00:00Z")), "2026-09-29");
  assert.equal(simulationDay(new Date("2026-01-15T05:30:00Z")), "2026-01-14");
  assert.equal(simulateBook(book([]), { ...options, asOf: "2026-01-05", lookbackDays: 14 }).from, "2025-12-23");
});

test("replays display their own dollars without duplicating live totals", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.01)]),
    day("2026-09-29", [pick(0.5)], [], { scan_id: "replay-1" }),
  ]), options);
  assert.equal(result.days[1].rank_money.pnl, 5_000);
  assert.equal(result.rank_money.pnl, 100);
  assert.equal(result.rank_days, 1);
  assert.equal(result.rank_stats.n, 1);
});

test("single-strategy and empty selections remain consistent", () => {
  const data = book([day("2026-09-29", [pick(0.01)], [pick(-0.01)])]);
  const rankOnly = simulateBook(data, { ...options, kind: "rank" });
  assert.equal(rankOnly.fit_money.pnl, null);
  assert.equal(rankOnly.days[0].fit.length, 0);
  const fitOnly = simulateBook(data, { ...options, kind: "fit" });
  assert.equal(fitOnly.rank_money.pnl, null);
  const empty = simulateBook(book([]), options);
  assert.equal(empty.rank_money.pnl, null);
  assert.equal(empty.fit_money.endingValue, null);
});
