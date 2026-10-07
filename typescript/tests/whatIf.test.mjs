import assert from "node:assert/strict";
import test from "node:test";
import {
  benchmarkDates, compareStrategyToBenchmark, DEFAULT_TRADE_SIZE, loadStrategyBenchmarks, TRADE_SIZE_OPTIONS,
  simulateBook, simulateOvernightHold, simulatePick, simulationDay, summarizeOvernightHold,
} from "../src/whatIf.js";

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
  kind: "both", fitTopK: 2, rankTopK: 2,
  tradeSizes: { rank: DEFAULT_TRADE_SIZE, fit: DEFAULT_TRADE_SIZE },
  lookbackDays: null, asOf: "2026-09-29",
};

test("default is $10K per trade, including losses and flat trades", () => {
  assert.equal(DEFAULT_TRADE_SIZE, 10_000);
  assert.deepEqual(simulatePick(pick(0.02), 10_000), { status: "scored", pnl: 200, endingValue: 10_200 });
  assert.deepEqual(simulatePick(pick(-0.01), 10_000), { status: "scored", pnl: -100, endingValue: 9_900 });
  assert.deepEqual(simulatePick(pick(0), 5_000), { status: "scored", pnl: 0, endingValue: 5_000 });
});

test("holding 50% to the observed next open changes only the separate comparison", () => {
  const row = { ...pick(0.02, { close_price: 102 }), hypothetical: simulatePick(pick(0.02), 10_000) };
  const actual = { ticker: "AAA", session: "2026-10-05", next_session: "2026-10-06",
    status: "observed", verified_close: 102, next_open: 103, reason: null };
  const half = simulateOvernightHold(row, actual, 10_000, 0.5);
  assert.deepEqual(half, { status: "observed", incrementalPnl: 50, endingValue: 10_250, includedInTotals: true });
  assert.equal(row.hypothetical.endingValue, 10_200);
  assert.deepEqual(summarizeOvernightHold([half], 200), {
    included: 1, observed: 1, incrementalPnl: 50, scenarioPnl: 250,
  });
});

test("pending next opens and skipped names cannot silently enter a basket total", () => {
  const row = { ...pick(0.02, { close_price: 102 }), hypothetical: simulatePick(pick(0.02), 10_000) };
  const pending = simulateOvernightHold(row, { status: "awaiting_next_open" }, 10_000, 0.3);
  assert.equal(pending.status, "awaiting_next_open");
  assert.deepEqual(summarizeOvernightHold([pending], 200), {
    included: 1, observed: 0, incrementalPnl: 0, scenarioPnl: null,
  });
  const skip = { ...row, hypothetical: simulatePick(pick(0.02, {
    news_blocks: true, news_flag: "corporate action",
  }), 10_000) };
  const observed = simulateOvernightHold(skip, { status: "observed", verified_close: 102, next_open: 103 }, 10_000, 1);
  assert.equal(observed.endingValue, 10_300);
  assert.equal(observed.includedInTotals, false);
  assert.equal(summarizeOvernightHold([observed], null).scenarioPnl, null);
  assert.equal(simulateOvernightHold(row, { status: "observed", verified_close: 101, next_open: 103 }, 10_000, 1).status, "price_mismatch");
});

test("shared trade-size choices support all requested amounts", () => {
  assert.deepEqual(TRADE_SIZE_OPTIONS, [3_000, 4_000, 5_000, 6_000, 8_000, 10_000, 15_000, 20_000]);
  assert.ok(TRADE_SIZE_OPTIONS.includes(DEFAULT_TRADE_SIZE));
  for (const amount of TRADE_SIZE_OPTIONS) {
    assert.deepEqual(simulatePick(pick(0.01), amount), {
      status: "scored", pnl: amount / 100, endingValue: amount + amount / 100,
    });
  }
});

test("hard skips show hypothetical end value without entering totals; holds can count", () => {
  for (const value of [null, NaN, Infinity]) {
    assert.equal(simulatePick(pick(value), 10_000).status, "pending");
  }
  assert.deepEqual(simulatePick(pick(0.5, { news_blocks: true, news_flag: "corporate action" }), 10_000), {
    status: "skipped", pnl: null, endingValue: 15_000,
  });
  assert.deepEqual(simulatePick(pick(null, { news_blocks: true, news_flag: "corporate action" }), 10_000), {
    status: "skipped", pnl: null, endingValue: null,
  });
  assert.deepEqual(simulatePick(pick(0.5, { news_blocks: true, news_flag: null }), 10_000), {
    status: "held", pnl: 5_000, endingValue: 15_000,
  });
  for (const amount of [0, -1, NaN, Infinity]) assert.throws(() => simulatePick(pick(0.01), amount), RangeError);
});

test("top 2 means $20K per strategy per day; strategies stay independent", () => {
  const data = book([day("2026-09-29", [pick(0.02), pick(-0.01, { rank: 2 }), pick(0.9, { rank: 3 })],
    [pick(0.03), pick(0.01, { rank: 2 })])]);
  const result = simulateBook(data, options);
  assert.deepEqual(result.rank_money, { pnl: 100, invested: 20_000, endingValue: 20_100, completed: 2, pending: 0 });
  assert.deepEqual(result.fit_money, { pnl: 400, invested: 20_000, endingValue: 20_400, completed: 2, pending: 0 });
  const half = simulateBook(data, { ...options, tradeSizes: { rank: 5_000, fit: 5_000 } });
  assert.equal(half.rank_money.pnl, 50);
  assert.equal(half.fit_money.pnl, 200);
  assert.equal(half.days[0].rank[0].hypothetical.endingValue, 5_100);
  assert.equal(data.days[0].rank[0].hypothetical, undefined); // Pure, no API-data mutation.
});

test("Rank $10K and Fit $5K reconcile separately at pick, day and period levels", () => {
  const data = book([
    day("2026-09-29", [pick(0.02), pick(-0.01, { rank: 2 })], [pick(0.03), pick(0.01, { rank: 2 })]),
    day("2026-09-28", [pick(-0.01)], [pick(-0.02)]),
  ]);
  const result = simulateBook(data, { ...options, tradeSizes: { rank: 10_000, fit: 5_000 } });
  assert.deepEqual(result.days[0].fit[0].hypothetical, { status: "scored", pnl: 150, endingValue: 5_150 });
  assert.deepEqual(result.days[0].fit_money, { pnl: 200, invested: 10_000, endingValue: 10_200, completed: 2, pending: 0 });
  assert.deepEqual(result.fit_money, { pnl: 100, invested: 15_000, endingValue: 15_100, completed: 3, pending: 0 });
  assert.deepEqual(result.rank_money, { pnl: 0, invested: 30_000, endingValue: 30_000, completed: 3, pending: 0 });
  const equalSizes = simulateBook(data, options);
  assert.deepEqual(result.rank_money, equalSizes.rank_money);
  assert.deepEqual(result.days.map((row) => row.rank), equalSizes.days.map((row) => row.rank));
  assert.deepEqual(result.fit_stats, equalSizes.fit_stats); // Sizing does not alter percentage statistics.
  const smallerRank = simulateBook(data, { ...options, tradeSizes: { rank: 5_000, fit: 5_000 } });
  assert.deepEqual(smallerRank.fit_money, result.fit_money);
  assert.deepEqual(smallerRank.days.map((row) => row.fit), result.days.map((row) => row.fit));
  assert.equal(smallerRank.days[0].rank_money.pnl, 50);
  assert.deepEqual(simulateBook(data, { ...options, kind: "fit", tradeSizes: { rank: 10_000, fit: 5_000 } }).fit_money, result.fit_money);
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
    pick(0.9, { news_blocks: true, news_flag: "corporate action" }), pick(null, { rank: 2 }), pick(0.1, { rank: 3 }),
  ])]), options);
  assert.equal(result.rank_money.pnl, null);
  assert.equal(result.rank_money.endingValue, null);
  assert.equal(result.rank_money.invested, 0);
  assert.equal(result.rank_money.pending, 1);
  assert.equal(result.rank_stats.n_avoid, 1);
  assert.equal(result.rank_days, 0);
  assert.equal(result.days[0].rank[0].hypothetical.endingValue, 19_000);
});

test("completed news holds contribute to totals while hard skips do not", () => {
  const result = simulateBook(book([day("2026-09-29", [
    pick(-0.02, { news_blocks: true, news_flag: null }),
    pick(0.5, { rank: 2, news_blocks: true, news_flag: "corporate action" }),
  ])]), options);
  assert.deepEqual(result.rank_money, { pnl: -200, invested: 10_000, endingValue: 9_800, completed: 1, pending: 0 });
  assert.equal(result.days[0].rank[1].hypothetical.endingValue, 15_000);
  assert.equal(result.rank_stats.n_scored, 1);
  assert.equal(result.rank_stats.n_avoid, 1);
  assert.deepEqual(benchmarkDates(result, "rank"), ["2026-09-29"]);
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

test("Rank and Fit SPY comparisons honor independent sizing, top K and active dates", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.01), pick(0.02, { rank: 2 }), pick(0.9, { rank: 3 })], [pick(0.01)]),
    day("2026-09-28", [pick(0.01)], [pick(null)]),
  ]), { ...options, tradeSizes: { rank: 10_000, fit: 5_000 } });
  assert.deepEqual(benchmarkDates(result, "rank"), ["2026-09-28", "2026-09-29"]);
  assert.deepEqual(benchmarkDates(result, "fit"), ["2026-09-29"]);
  const spy = {
    returns: { "2026-09-28": -0.01, "2026-09-29": 0.02 },
    hold: { start: "2026-09-28", end: "2026-09-29", pct: 0.03 },
  };
  const rank = compareStrategyToBenchmark(result, "rank", spy);
  assert.deepEqual(rank.intraday, { pnl: 300, pct: 0.01 });
  assert.equal(rank.intradayCapital, 30_000);
  assert.equal(rank.holdCapital, 15_000);
  assert.ok(Math.abs(rank.hold.pct - 0.0197) < 1e-12);
  assert.equal(rank.hold.pnl, 295.5);
  const fit = compareStrategyToBenchmark(result, "fit", spy);
  assert.deepEqual(fit.intraday, { pnl: 100, pct: 0.02 });
  assert.equal(fit.holdCapital, 5_000);
  assert.deepEqual(fit.hold, { pnl: null, pct: null }); // This response covers Rank's different interval.
});

test("skips, pending closes, and replay rows do not create SPY exposure", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.01, { news_blocks: true, news_flag: "corporate action" }), pick(null, { rank: 2 })]),
    day("2026-09-28", [pick(0.01), pick(0.02, { rank: 2 })]),
    day("2026-09-28", [pick(0.5)], [], { scan_id: "replay" }),
  ]), options);
  assert.deepEqual(benchmarkDates(result, "rank"), ["2026-09-28"]);
  const rank = compareStrategyToBenchmark(result, "rank", {
    returns: { "2026-09-28": -0.01, "2026-09-29": 0.5 },
    hold: { start: "2026-09-28", end: "2026-09-29", pct: 0.1 },
  });
  assert.equal(rank.activeDays, 1);
  assert.deepEqual(rank.intraday, { pnl: -200, pct: -0.01 });
  assert.equal(rank.hold.pnl, null);
});

test("missing SPY sessions are excluded from the intraday denominator, not treated as flat", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.01), pick(0.01, { rank: 2 })]),
    day("2026-09-25", [pick(0.01)]),
  ]), options);
  const rank = compareStrategyToBenchmark(result, "rank", {
    returns: { "2026-09-25": 0.01 },
    hold: { start: "2026-09-25", end: "2026-09-29", pct: -0.02 },
  });
  assert.equal(rank.matchedDays, 1);
  assert.equal(rank.activeDays, 2);
  assert.equal(rank.intradayCapital, 10_000);
  assert.deepEqual(rank.intraday, { pnl: 100, pct: 0.01 });
  assert.equal(rank.holdCapital, 15_000);
  assert.ok(Math.abs(rank.hold.pct - (-0.0102)) < 1e-12);
  assert.equal(rank.hold.pnl, -153); // First day's open, then continuous hold through intervening days.
  assert.deepEqual(compareStrategyToBenchmark(result, "rank", null).intraday, { pnl: null, pct: null });
});

test("missing or shifted SPY hold interval is never presented as the selected interval", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.01)]), day("2026-09-28", [pick(0.01)]),
  ]), options);
  for (const hold of [null, { start: "2026-09-27", end: "2026-09-29", pct: 0.01 },
    { start: "2026-09-28", end: "2026-09-29", pct: NaN }]) {
    const comparison = compareStrategyToBenchmark(result, "rank", { returns: {}, hold });
    assert.deepEqual(comparison.hold, { pnl: null, pct: null });
    assert.equal(comparison.holdFrom, null);
  }
});

test("duplicate tickers remain distinct selected picks; invalid SPY returns never produce NaN", () => {
  const result = simulateBook(book([
    day("2026-09-29", [pick(0.01), pick(0.02, { rank: 2 })]),
  ]), options);
  assert.equal(result.rank_money.completed, 2);
  const missing = compareStrategyToBenchmark(result, "rank", { returns: { "2026-09-29": NaN } });
  assert.equal(missing.intraday.pnl, null);
  assert.equal(missing.intraday.pct, null);
  assert.equal(missing.intradayCapital, 0);
  const flat = compareStrategyToBenchmark(result, "rank", { returns: { "2026-09-29": 0 } });
  assert.deepEqual(flat.intraday, { pnl: 0, pct: 0 });
});

test("a one-day SPY buy-and-hold is that day's open-to-close return", () => {
  const result = simulateBook(book([day("2026-09-29", [pick(0.01)])]), options);
  const hold = { start: "2026-09-29", end: "2026-09-29", pct: 0 };
  const comparison = compareStrategyToBenchmark(result, "rank", {
    returns: { "2026-09-29": 0.025 }, hold,
  });
  assert.equal(comparison.hold.pnl, 250);
  assert.ok(Math.abs(comparison.hold.pct - 0.025) < 1e-12);
  assert.equal(comparison.holdCapital, 10_000);
  assert.deepEqual(compareStrategyToBenchmark(result, "rank", { returns: {}, hold }).hold,
    { pnl: null, pct: null }); // First open is unknown.
});

test("benchmark loads deduplicate matching date sets and keep strategy failures independent", async () => {
  const calls = [];
  const fetcher = async (dates) => {
    calls.push(dates.join(","));
    if (dates[0] === "2026-09-28") throw new Error("old API: 404");
    return { returns: { "2026-09-29": 0.01 } };
  };
  const result = await loadStrategyBenchmarks(["2026-09-28"], ["2026-09-29"], fetcher);
  assert.match(result.rank.error, /404/);
  assert.equal(result.rank.response, null);
  assert.equal(result.fit.error, null);
  assert.equal(result.fit.response.returns["2026-09-29"], 0.01);
  const shared = await loadStrategyBenchmarks(["2026-09-29"], ["2026-09-29"], fetcher);
  assert.equal(shared.rank, shared.fit);
  assert.deepEqual(calls, ["2026-09-28", "2026-09-29", "2026-09-29"]);
});

test("historical benchmark cache reuses successes, retries failures, and never caches today", async () => {
  const cache = new Map();
  let calls = 0;
  const fetcher = async () => {
    calls += 1;
    if (calls === 1) throw new Error("temporary");
    return { returns: { "2026-09-28": 0.01, "2026-09-29": 0.02 } };
  };
  const first = await loadStrategyBenchmarks(["2026-09-28"], [], fetcher, cache, "2026-09-29");
  assert.match(first.rank.error, /temporary/);
  await loadStrategyBenchmarks(["2026-09-28"], [], fetcher, cache, "2026-09-29");
  await loadStrategyBenchmarks(["2026-09-28"], [], fetcher, cache, "2026-09-29");
  assert.equal(calls, 2);
  await loadStrategyBenchmarks(["2026-09-29"], [], fetcher, cache, "2026-09-29");
  await loadStrategyBenchmarks(["2026-09-29"], [], fetcher, cache, "2026-09-29");
  assert.equal(calls, 4);
});
