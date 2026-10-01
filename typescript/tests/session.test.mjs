import assert from "node:assert/strict";
import test from "node:test";
import { isCashSessionToday, isChicagoCalendarToday, scanHasClosed } from "../src/session.js";

test("Wednesday's saved scan stays visible after Eastern midnight but before Chicago midnight", () => {
  const lateWednesday = new Date("2026-10-01T04:15:00Z"); // 23:15 CT, 00:15 ET
  assert.equal(isCashSessionToday("2026-09-30", lateWednesday), false);
  assert.equal(isChicagoCalendarToday("2026-09-30", lateWednesday), true);
  assert.equal(scanHasClosed("2026-09-30", lateWednesday), true);
});

test("yesterday's scan is not treated as current on Thursday morning", () => {
  const thursdayOpen = new Date("2026-10-01T13:30:00Z"); // 08:30 CT
  assert.equal(isChicagoCalendarToday("2026-09-30", thursdayOpen), false);
  assert.equal(isCashSessionToday("2026-09-30", thursdayOpen), false);
  assert.equal(scanHasClosed("2026-10-01", thursdayOpen), false);
});
