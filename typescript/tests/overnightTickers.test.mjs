import assert from "node:assert/strict";
import test from "node:test";
import { tradedTickerSuggestions } from "../src/overnightTickers.js";

test("open positions come first and aggregate open lots", () => {
  const positions = [
    { ticker: "frmi", shares: 500, closed: false },
    { ticker: "FRMI", shares: 300, closed: false },
    { ticker: "MMED", shares: 100, closed: true },
  ];
  const trades = [
    { ticker: "FSLY", executed_at: "2026-10-12T17:09:42Z" },
    { ticker: "FRMI", executed_at: "2026-10-12T15:56:28Z" },
    { ticker: "CRGY", executed_at: "2026-10-09T19:41:38Z" },
  ];
  assert.deepEqual(tradedTickerSuggestions(positions, trades), [
    { ticker: "FRMI", detail: "Open · 800 shares" },
    { ticker: "FSLY", detail: "Recent trade" },
    { ticker: "CRGY", detail: "Recent trade" },
  ]);
});

test("recent trade symbols are unique and ordered by execution time", () => {
  const trades = [
    { ticker: "HPQ", executed_at: "2026-10-09T18:27:48Z" },
    { ticker: "FSLY", executed_at: "2026-10-12T17:09:42Z" },
    { ticker: "hpq", executed_at: "2026-10-09T18:27:19Z" },
  ];
  assert.deepEqual(tradedTickerSuggestions([], trades).map(({ ticker }) => ticker), ["FSLY", "HPQ"]);
});
