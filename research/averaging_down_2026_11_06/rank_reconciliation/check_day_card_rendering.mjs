/** Render the production day-card component and reconcile its displayed P&L. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const repo = process.cwd();
const ui = resolve(repo, 'typescript');
const root = new URL('./', import.meta.url);
const read = (name) => JSON.parse(readFileSync(new URL(name, root), 'utf8'));
const previous = process.argv.includes('--previous');
const { createServer } = await import(pathToFileURL(resolve(ui, 'node_modules/vite/dist/node/index.js')));
const { default: React } = await import(pathToFileURL(resolve(ui, 'node_modules/react/index.js')));
const { renderToStaticMarkup } = await import(pathToFileURL(resolve(ui, 'node_modules/react-dom/server.node.js')));
const server = await createServer({
  root: ui, configFile: false, appType: 'custom', logLevel: 'error',
  esbuild: { jsx: 'automatic' },
  server: { middlewareMode: true, hmr: false, watch: null },
  plugins: [{
    name: 'expose-day-card-for-verification', enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('/src/components/WhatIf.tsx')) return;
      const source = previous
        ? execFileSync('git', ['show', 'HEAD:typescript/src/components/WhatIf.tsx'], { cwd: repo, encoding: 'utf8' })
        : code;
      // Export the existing component only in this in-memory verification build.
      return source.replace('function DayCard(', 'export function DayCard(');
    },
  }],
});

try {
  const { DayCard } = await server.ssrLoadModule('/src/components/WhatIf.tsx');
  const { simulateBook, summarizeSelectedOvernightHistory } = await server.ssrLoadModule('/src/whatIf.ts');
  const tradeSizes = { rank: 10000, fit: 10000 };
  const options = { kind: 'both', rankTopK: 2, fitTopK: 2, applyNewsSkips: false,
    tradeSizes, lookbackDays: null, asOf: '2026-11-06' };
  const render = (day, actuals, mode, strategy = 'rank', portion = 100, chosen = []) => {
    const html = renderToStaticMarkup(React.createElement(DayCard, {
      day, kind: strategy, rankAsked: 1, fitAsked: 1, tradeSizes, actuals,
      overnightPortion: portion, exitMode: mode,
      custom: { rank: new Set(chosen), fit: new Set(chosen) }, onToggleHold() {},
    }));
    const summary = html.match(/<summary>([\s\S]*?)<\/summary>/)[1].replace(/<[^>]+>/g, '');
    const amount = summary.match(/([▲▼])\s*\$([\d,.]+)/);
    return { text: summary, pnl: amount ? (amount[1] === '▼' ? -1 : 1) * Number(amount[2].replaceAll(',', '')) : null };
  };
  const failures = [];
  let checks = 0;
  function check(label, result, expected) {
    checks += 1;
    if (result.pnl !== expected) failures.push({ label, expected, ...result });
  }
  const loser = { ticker: 'AAA', rank: 1, open_price: 100, close_price: 90, session_return: -0.1, predicted: 0.01 };
  const winner = { ticker: 'BBB', rank: 2, open_price: 100, close_price: 105, session_return: 0.05, predicted: 0.01 };
  const synthetic = (rows) => simulateBook({ days: [{ as_of: '2026-11-05', scan_id: '', rank: rows, fit: rows }] }, options).days[0];
  const actuals = new Map([
    ['AAA', { status: 'observed', verified_close: 90, next_open: 95 }],
    ['BBB', { status: 'observed', verified_close: 105, next_open: 107 }],
  ]);
  for (const strategy of ['rank', 'fit']) {
    check(`${strategy}: sell at close`, render(synthetic([loser]), actuals, 'close', strategy), -1000);
    check(`${strategy}: overnight recovery remains a loss`, render(synthetic([loser]), actuals, 'down', strategy), -500);
    check(`${strategy}: down-only includes close-sale winners`, render(synthetic([loser, winner]), actuals, 'down', strategy), 0);
    check(`${strategy}: all at next open`, render(synthetic([loser, winner]), actuals, 'all', strategy), 200);
    check(`${strategy}: custom next-open sale`, render(synthetic([loser, winner]), actuals, 'custom', strategy, 100, ['BBB']), -300);
    check(`${strategy}: partial overnight holding`, render(synthetic([loser, winner]), actuals, 'down', strategy, 25), -375);
    check(`${strategy}: pending next open`, render(synthetic([loser]), new Map([
      ['AAA', { status: 'awaiting_next_open' }],
    ]), 'down', strategy), null);
  }
  const data = read('live_repaired_api_snapshot.json');
  const docs = read('overnight_actuals_snapshot.json');
  const byDay = new Map(Object.entries(docs).map(([date, doc]) => [date, new Map(doc.rows.map(row => [row.ticker, row]))]));
  const book = simulateBook(data, { ...options, kind: 'rank', rankTopK: 1 });
  const expectedRows = new Map(read('rank1_significance_rows.json').map(row => [row.date, row]));
  const rows = book.days.filter(day => !day.scan_id && day.rank.length).map(day => {
    const displayed = render(day, byDay.get(day.as_of), 'down');
    check(day.as_of, displayed, expectedRows.get(day.as_of).pnl);
    return { date: day.as_of, ticker: day.rank[0].ticker, ...displayed };
  });
  const total = Math.round(rows.reduce((sum, row) => sum + row.pnl, 0) * 100) / 100;
  const headline = summarizeSelectedOvernightHistory(book.days, byDay, 'rank', 10000, 1, 'down');
  const report = { previous, checks, failures, total, headline: headline.pnl,
    days: rows.length, wins: rows.filter(row => row.pnl > 0).length,
    losses: rows.filter(row => row.pnl < 0).length, rows };
  writeFileSync(new URL(previous ? 'day_card_render_before.json' : 'day_card_render_after.json', root), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, rows: rows.filter(row => ['2026-09-25', '2026-11-05'].includes(row.date)) }, null, 2));
  if (!previous) {
    assert.deepEqual(failures, []);
    assert.equal(total, headline.pnl);
    assert.equal(total, 2708.87);
    assert.equal(report.wins, 25);
    assert.equal(report.losses, 11);
  }
} finally {
  await server.close();
}
