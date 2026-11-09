/** Reproduce the screenshot and isolate corrections without changing stored data. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { simulateBook, summarizeSelectedOvernightHistory } from './compiled/whatIf.mjs';

const root = new URL('./', import.meta.url);
const read = (name) => JSON.parse(readFileSync(new URL(name, root), 'utf8'));
const original = read('paper_book_snapshot.json');
const daily = new Map(read('daily_price_snapshot.json').map((row) => [
  `${row.date}:${row.ticker}`, row.payload,
]));
const actuals = new Map(Object.entries(read('overnight_actuals_snapshot.json')).map(([day, data]) => [
  day, new Map(data.rows.map((row) => [row.ticker, row])),
]));
const stale = new Set(['2026-09-21:PS', '2026-09-22:HHH', '2026-09-23:TRLV']);
const scenarios = [
  ['unchanged', () => false, false, false],
  ['three_confirmed_stale_opens', (key) => stale.has(key), true, false],
  ['all_verified_opens', () => true, true, false],
  ['all_verified_closes', () => true, false, true],
  ['all_verified_opens_and_closes', () => true, true, true],
];
const results = scenarios.map(([scenario, selected, useOpen, useClose]) => {
  const data = structuredClone(original);
  for (const day of data.days) {
    if (day.scan_id) continue;
    for (const row of day.rank) {
      const key = `${day.as_of}:${row.ticker}`;
      const price = daily.get(key);
      if (!price || !selected(key)) continue;
      assert.equal(price.status, 'OK');
      assert.equal(price.from, day.as_of);
      assert.equal(price.symbol, row.ticker);
      if (useOpen) row.open_price = price.open;
      if (useClose) row.close_price = price.close;
      row.session_return = row.close_price / row.open_price - 1;
    }
  }
  const book = simulateBook(data, {
    kind: 'rank', rankTopK: 2, applyNewsSkips: false,
    tradeSizes: { rank: 10000, fit: 10000 }, lookbackDays: null, asOf: '2026-11-06',
  });
  const history = summarizeSelectedOvernightHistory(book.days, actuals, 'rank', 10000, 1, 'down');
  assert.equal(history.matchedDays, 35);
  const completedPositions = history.matchedDays * 2;
  return {
    scenario, ...history, completedPositions,
    averagePerPosition: history.pnl / completedPositions,
    periodGain: history.pnl / 20000,
    differenceFromScreenshot: Math.round((history.pnl - 5882.93) * 100) / 100,
  };
});
assert.equal(results[0].pnl, 5882.93);
assert.equal(results.at(-1).pnl, 2454.04);
writeFileSync(new URL('isolated_price_impact.json', root), JSON.stringify(results, null, 2) + '\n');
console.log(JSON.stringify(results, null, 2));
