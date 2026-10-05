import assert from "node:assert/strict";
import test from "node:test";
import { newsBlocksBuy } from "../src/newsSkip.js";

test("incomplete reviews are holds even without a headline flag", () => {
  for (const status of ["degraded", "error", "not_checked", "unknown"]) {
    assert.equal(newsBlocksBuy(null, 11, 10, status), true);
  }
  assert.equal(newsBlocksBuy(null, 11, 10, "no_news"), false);
  assert.equal(newsBlocksBuy(null, 11, 10, "complete"), false);
});

test("spin-offs and released clinical data skip regardless of the gap", () => {
  assert.equal(newsBlocksBuy("corporate action: CTVA spin-off", 11, 10, "complete"), true);
  assert.equal(newsBlocksBuy("clinical readout: NKTR Phase 2b data", 11, null, "complete"), true);
  assert.equal(newsBlocksBuy("ordinary headline", 11, 10, "complete"), false);
  assert.equal(newsBlocksBuy("ordinary headline", 9, 10, "complete"), true);
});
