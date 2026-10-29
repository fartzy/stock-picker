import assert from "node:assert/strict";
import test from "node:test";
import { loadHistoryDefaults, parseHistoryDefaults, saveHistoryDefaults } from "../src/appSettings.js";

test("new and invalid preferences keep weeks closed and the latest month open", () => {
  for (const raw of [null, "broken", "null", "[]", "true", "{}"])
    assert.deepEqual(parseHistoryDefaults(raw), { keepRecentWeekOpen: false, keepRecentMonthOpen: true });
});

test("valid false values survive and malformed fields fall back independently", () => {
  assert.deepEqual(parseHistoryDefaults('{"keepRecentWeekOpen":true,"keepRecentMonthOpen":false}'),
    { keepRecentWeekOpen: true, keepRecentMonthOpen: false });
  assert.deepEqual(parseHistoryDefaults('{"keepRecentWeekOpen":"false","keepRecentMonthOpen":false,"extra":42}'),
    { keepRecentWeekOpen: false, keepRecentMonthOpen: false });
});

test("unavailable browser storage does not prevent rendering or changing preferences", () => {
  assert.deepEqual(loadHistoryDefaults(), { keepRecentWeekOpen: false, keepRecentMonthOpen: true });
  assert.equal(saveHistoryDefaults({ keepRecentWeekOpen: true, keepRecentMonthOpen: false }), false);
});
