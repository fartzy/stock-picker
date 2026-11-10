import assert from "node:assert/strict";
import test from "node:test";
import { entryClock, entryEvidenceMessage, entryPrice, entryProbability, entryReturn, entryScoreAvailable, parseEntryTickers } from "../src/intradayEntry.js";
import { predictIntradayEntry, trainIntradayEntry } from "../src/api.js";

test("manual selection keeps ordering and accepts class-share symbols without duplicate requests", () => {
  assert.deepEqual(parseEntryTickers(" ttmi, LUMN; TTMI\nBRK.B BRK-B "), { tickers: ["TTMI", "LUMN", "BRK.B", "BRK-B"], error: null });
  assert.match(parseEntryTickers(" ").error, /Enter a ticker/);
  assert.match(parseEntryTickers("TTMI, $LUMN").error, /Check the ticker/);
});

test("refresh enforces twenty unique selections", () => {
  const twenty = Array.from({ length: 20 }, (_, index) => `S${index}`).join(",");
  assert.equal(parseEntryTickers(`${twenty},S0,S1`).error, null);
  assert.match(parseEntryTickers(`${twenty},S20`).error, /up to 20/);
});

test("unavailable model outputs never become fabricated zero probabilities", () => {
  for (const value of [null, undefined, Number.NaN, Infinity, -0.01, 1.01]) assert.equal(entryProbability(value), "—");
  assert.equal(entryProbability(0), "0%");
  assert.equal(entryProbability(0.72), "72%");
  const valid = { status: "ok", probability_profit: 0, expected_return: 0, probability_loss_2pct: 0 };
  assert.equal(entryScoreAvailable(valid), true);
  assert.equal(entryScoreAvailable({ ...valid, status: "unavailable" }), false);
  assert.equal(entryScoreAvailable({ ...valid, expected_return: null }), false);
  assert.equal(entryScoreAvailable({ ...valid, probability_profit: Number.NaN }), false);
});

test("returns are fractions and timestamp rendering handles daylight saving", () => {
  assert.equal(entryReturn(0.0301), "+3.01%");
  assert.equal(entryReturn(-0.02), "-2.00%");
  assert.equal(entryReturn(null), "—");
  assert.equal(entryPrice(73.655), "$73.655");
  assert.equal(entryPrice(0), "—");
  assert.equal(entryClock("2026-11-10T15:40:00Z"), "9:40:00 AM");
  assert.equal(entryClock("2026-07-10T14:40:00Z"), "9:40:00 AM");
  assert.equal(entryClock("bad timestamp"), "—");
});

test("evidence sentence follows held-out Rank #1 error, including ties and missing comparisons", () => {
  assert.match(entryEvidenceMessage({ brier: 0.25639787, baseline_brier: 0.24583812 }), /did not beat/);
  assert.match(entryEvidenceMessage({ brier: 0.24, baseline_brier: 0.24 }), /did not beat/);
  assert.match(entryEvidenceMessage({ brier: 0.23, baseline_brier: 0.24 }), /odds beat/);
  assert.match(entryEvidenceMessage({ brier: null, baseline_brier: 0.24 }), /not available/);
  assert.match(entryEvidenceMessage(null), /not available/);
});

test("refresh posts only selected tickers and never starts training", async () => {
  const calls = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (path, options) => {
    calls.push({ path, ...options });
    return new Response(JSON.stringify({ predictions: [] }), { status: 200 });
  };
  try {
    await predictIntradayEntry(["TTMI", "LUMN"]);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].path, "/api/intraday-entry/predict");
    assert.equal(calls[0].method, "POST");
    assert.deepEqual(JSON.parse(calls[0].body), { tickers: ["TTMI", "LUMN"] });
    await trainIntradayEntry();
    assert.equal(calls[1].path, "/api/intraday-entry/training");
    assert.equal(calls[1].method, "POST");
  } finally { globalThis.fetch = original; }
});

test("server missing-data errors reach the panel instead of displaying predictions", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Train the entry model first." }), { status: 409 });
  try { await assert.rejects(predictIntradayEntry(["TTMI"]), /Train the entry model first/); }
  finally { globalThis.fetch = original; }
});
