import assert from "node:assert/strict";
import test from "node:test";
import { loadHistoryDefaults, parseHistoryDefaults, saveHistoryDefaults } from "../src/appSettings.js";

test("new and invalid preferences keep weeks closed and the latest month open", () => {
  for (const raw of [null, "broken", "null", "[]", "true", "{}"])
    assert.deepEqual(parseHistoryDefaults(raw), { grouping: "months", keepRecentWeekOpen: false, keepRecentMonthOpen: true });
});

test("valid false values survive and malformed fields fall back independently", () => {
  assert.deepEqual(parseHistoryDefaults('{"keepRecentWeekOpen":true,"keepRecentMonthOpen":false}'),
    { grouping: "months", keepRecentWeekOpen: true, keepRecentMonthOpen: false });
  assert.deepEqual(parseHistoryDefaults('{"keepRecentWeekOpen":"false","keepRecentMonthOpen":false,"extra":42}'),
    { grouping: "months", keepRecentWeekOpen: false, keepRecentMonthOpen: false });
});

test("unavailable browser storage does not prevent rendering or changing preferences", () => {
  assert.deepEqual(loadHistoryDefaults(), { grouping: "months", keepRecentWeekOpen: false, keepRecentMonthOpen: true });
  assert.equal(saveHistoryDefaults({ grouping: "months", keepRecentWeekOpen: true, keepRecentMonthOpen: false }), false);
});


test("all default views persist; old or unsupported values migrate to months", () => {
  for (const grouping of ["days", "weeks", "months"]) {
    assert.equal(parseHistoryDefaults(JSON.stringify({ grouping })).grouping, grouping);
  }
  for (const grouping of ["years", "", null, false, 5]) {
    assert.equal(parseHistoryDefaults(JSON.stringify({ grouping })).grouping, "months");
  }
  assert.deepEqual(parseHistoryDefaults('{"keepRecentWeekOpen":true,"keepRecentMonthOpen":false}'),
    { grouping: "months", keepRecentWeekOpen: true, keepRecentMonthOpen: false });
});
