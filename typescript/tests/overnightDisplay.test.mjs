import assert from "node:assert/strict";
import test from "node:test";
import { overnightDirection, overnightDollarText, overnightGapText } from "../src/overnightDisplay.js";

test("a tiny nonzero gap does not render as Up +0.00% and $0.00", () => {
  const direction = overnightDirection(0.000001, 0.0001);
  assert.equal(direction, "flat");
  assert.equal(overnightGapText(0.000001, direction), "≈0%");
  assert.equal(overnightDollarText(0.0001), "<$0.01");
});

test("a sub-0.01% gap can still show direction when its dollar move is visible", () => {
  const direction = overnightDirection(-0.00001, -0.01);
  assert.equal(direction, "down");
  assert.equal(overnightGapText(-0.00001, direction), "<0.01%");
  assert.equal(overnightDollarText(-0.01), "-$0.01");
});

test("ordinary signed moves retain their direction and precision", () => {
  assert.equal(overnightDirection(0.0005, 0.06), "up");
  assert.equal(overnightGapText(0.0005, "up"), "+0.05%");
  assert.equal(overnightDollarText(0.06), "+$0.06");
  assert.equal(overnightDirection(0, 0), "flat");
  assert.equal(overnightGapText(0, "flat"), "0.00%");
});
