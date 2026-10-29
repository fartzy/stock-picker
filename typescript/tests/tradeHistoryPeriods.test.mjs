import assert from "node:assert/strict";
import test from "node:test";
import { groupHistoryPeriods, groupMonthWeeks, isWeekToDate } from "../src/tradeHistoryPeriods.js";

const days = (...dates) => dates.map((day, index) => [day, [{ pnl: index + 1 }]]);

test("calendar months partition a crossing week without losing or double-counting days", () => {
  const input = days("2026-09-28", "2026-10-02", "2026-09-30", "2026-10-01");
  const months = groupHistoryPeriods(input, "months");
  assert.deepEqual(months.map((p) => p.label), ["October 2026", "September 2026"]);
  assert.deepEqual(months.map((p) => p.days.map(([day]) => day)), [
    ["2026-10-02", "2026-10-01"], ["2026-09-30", "2026-09-28"],
  ]);
  assert.equal(months.flatMap((p) => p.days).flatMap(([, rows]) => rows).reduce((sum, r) => sum + r.pnl, 0), 10);
  const weeks = groupHistoryPeriods(input, "weeks");
  assert.equal(weeks.length, 1);
  assert.equal(weeks[0].label, "Week Sep 28–Oct 2");
  assert.equal(weeks[0].days.length, 4);
  const children = groupMonthWeeks(months[0]);
  assert.deepEqual(children.flatMap((p) => p.days), months[0].days);
  assert.equal(children[0].label, "Week Oct 1–Oct 2");
  const september = groupMonthWeeks(months[1]);
  assert.equal(september[0].label, "Week Sep 28–Sep 30");
  assert.notEqual(children[0].key, september[0].key);
  assert.notEqual(children[0].key, weeks[0].key);
  assert.deepEqual(input.map(([day]) => day), ["2026-09-28", "2026-10-02", "2026-09-30", "2026-10-01"]);
});

test("month weeks preserve all lots and use calendar boundaries, not sparse trade dates", () => {
  const input = days("2026-10-29", "2026-10-26", "2026-10-13", "2026-10-02");
  const [month] = groupHistoryPeriods(input, "months");
  const weeks = groupMonthWeeks(month);
  assert.deepEqual(weeks.map((w) => w.label), [
    "Week Oct 26–Oct 30", "Week Oct 12–Oct 16", "Week Oct 1–Oct 2",
  ]);
  assert.deepEqual(weeks.flatMap((w) => w.days), month.days);
  assert.equal(new Set(weeks.flatMap((w) => w.days.map(([day]) => day))).size, input.length);
});

test("partial weeks handle year transitions and leap-month endings", () => {
  const months = groupHistoryPeriods(days("2025-12-31", "2026-01-02", "2024-02-29"), "months");
  assert.deepEqual(months.flatMap((month) => groupMonthWeeks(month).map((w) => w.label)), [
    "Week Jan 1–Jan 2", "Week Dec 29–Dec 31", "Week Feb 26–Feb 29",
  ]);
  assert.deepEqual(groupMonthWeeks({ key: "months:2026-01-01", label: "January 2026", days: [] }), []);
});

test("months remain distinct across years and include leap day", () => {
  assert.deepEqual(groupHistoryPeriods(days("2025-01-03", "2026-01-02", "2024-02-29"), "months").map((p) => p.label),
    ["January 2026", "January 2025", "February 2024"]);
  assert.deepEqual(groupHistoryPeriods([], "months"), []);
});

test("WTD starts Monday, crosses year boundaries, and excludes future dates", () => {
  assert.equal(isWeekToDate("2025-12-29", "2026-01-01"), true);
  assert.equal(isWeekToDate("2025-12-28", "2026-01-01"), false);
  assert.equal(isWeekToDate("2026-01-02", "2026-01-01"), false);
  assert.equal(isWeekToDate("2026-10-26", "2026-11-01"), true);
  assert.equal(isWeekToDate("2026-10-30", "2026-11-02"), false);
  assert.equal(isWeekToDate("2026-11-02", "2026-11-02"), true);
});
