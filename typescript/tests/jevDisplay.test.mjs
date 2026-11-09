import assert from "node:assert/strict";
import { jevPercent, jevProbability, jevPriceBasis, jevTime, JEV_QUESTION_LABELS } from "../src/jevDisplay.js";

assert.equal(jevProbability(0.8), "80.0%");
assert.equal(jevProbability(1.1), "Not available");
assert.equal(jevProbability(Number.NaN), "Not available");
assert.equal(jevProbability(null), "Not available");
assert.equal(jevPercent(-0.03), "-3.0%");
assert.equal(jevPriceBasis("observed_trade"), "Observed last trade");
assert.equal(jevPriceBasis("session_close"), "Actual session close, observed later");
assert.equal(jevTime("bad time"), "Time unavailable");
assert.equal(JEV_QUESTION_LABELS.buy_today, "Close above the morning open");
assert.equal(JEV_QUESTION_LABELS.net_positive_by_close, "Profit after costs by close");
