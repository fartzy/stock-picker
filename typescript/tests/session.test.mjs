import assert from "node:assert/strict";
import test from "node:test";
import { expireQuotePrefill, isCashSessionToday, isChicagoCalendarToday, isFreshCashSessionQuote, scanHasClosed } from "../src/session.js";

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

test("current quote use ends at the 16:00 ET bell even while a recent quote remains displayed", () => {
  const quote = {
    observed_at: "2026-10-06T15:55:00-04:00",
    session_open_at: "2026-10-06T09:30:00-04:00",
    session_close_at: "2026-10-06T16:00:00-04:00",
  };
  const prefill = { value: "5.915", quote };
  assert.equal(isFreshCashSessionQuote(quote, Date.parse("2026-10-06T15:59:59-04:00")), true);
  assert.strictEqual(expireQuotePrefill(prefill, Date.parse("2026-10-06T15:59:59-04:00")), prefill);
  assert.equal(isFreshCashSessionQuote(quote, Date.parse("2026-10-06T16:00:00-04:00")), false);
  assert.deepEqual(expireQuotePrefill(prefill, Date.parse("2026-10-06T16:00:00-04:00")), { value: "", quote: null });
  assert.deepEqual(expireQuotePrefill(prefill, Date.parse("2026-10-06T16:05:00-04:00")), { value: "", quote: null });
  const manuallyEdited = { value: "5.915", quote: null };
  assert.strictEqual(expireQuotePrefill(manuallyEdited, Date.parse("2026-10-06T16:05:00-04:00")), manuallyEdited);
  assert.equal(isFreshCashSessionQuote({ ...quote, observed_at: "2026-10-06T16:04:00-04:00" }, Date.parse("2026-10-06T16:05:00-04:00")), false);
  assert.equal(isFreshCashSessionQuote({ ...quote, observed_at: "2026-10-06T09:29:00-04:00" }, Date.parse("2026-10-06T09:30:00-04:00")), false);
});

test("exchange-provided early close expires an untouched quote at 13:00 ET", () => {
  const quote = {
    observed_at: "2026-11-27T12:55:00-05:00",
    session_open_at: "2026-11-27T09:30:00-05:00",
    session_close_at: "2026-11-27T13:00:00-05:00",
  };
  assert.equal(isFreshCashSessionQuote(quote, Date.parse("2026-11-27T12:59:59-05:00")), true);
  assert.deepEqual(expireQuotePrefill({ value: "42.10", quote }, Date.parse("2026-11-27T13:00:00-05:00")), { value: "", quote: null });
  assert.equal(isFreshCashSessionQuote({ ...quote, observed_at: "2026-11-27T13:04:00-05:00" }, Date.parse("2026-11-27T13:05:00-05:00")), false);
});
