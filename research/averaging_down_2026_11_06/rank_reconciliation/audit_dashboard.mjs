/** Read-only replay of the production What If calculations on frozen API data. */
import { readFileSync, writeFileSync } from 'node:fs';
import {
  simulateBook, summarizeSelectedOvernightHistory, summarizeOvernightExit,
  simulateOvernightHold, overnightTradeReturn, compareStrategyToBenchmark,
  returnOnAverageDailyCapital,
} from './compiled/whatIf.mjs';

const root = new URL('./', import.meta.url);
const read = (name) => JSON.parse(readFileSync(new URL(name, root), 'utf8'));
const write = (name, value) => writeFileSync(new URL(name, root), JSON.stringify(value, null, 2));
const raw = read('paper_book_snapshot.json');
const actualDocs = read('overnight_actuals_snapshot.json');
const actuals = new Map(Object.entries(actualDocs).map(([date, doc]) => [date, new Map(doc.rows.map(r => [r.ticker, r]))]));
const options = {
  kind: 'rank', rankTopK: 2, applyNewsSkips: false,
  tradeSizes: { rank: 10000, fit: 10000 }, lookbackDays: null, asOf: '2026-11-06',
};
const book = simulateBook(raw, options);
const history = summarizeSelectedOvernightHistory(book.days, actuals, 'rank', 10000, 1, 'down');
const capital = compareStrategyToBenchmark(book, 'rank', null).holdCapital;
const live = book.days.filter(d => !d.scan_id && d.rank.length);
const rows = [];
const days = [];
for (const day of live) {
  const observed = actuals.get(day.as_of) ?? new Map();
  const exit = summarizeOvernightExit(day.rank, observed, 10000, day.rank_money, 1, 'down');
  days.push({ date: day.as_of, complete: exit.pnl !== null, close_pnl: day.rank_money.pnl, ...exit });
  for (const row of day.rank) {
    const held = row.session_return < 0;
    const overnight = simulateOvernightHold(row, observed.get(row.ticker), 10000, 1);
    const finalReturn = held ? overnightTradeReturn(row, observed.get(row.ticker), 10000, 1) : row.session_return;
    rows.push({
      date: day.as_of, rank: row.rank, ticker: row.ticker,
      entry_price: row.open_price, close_price: row.close_price,
      next_open: observed.get(row.ticker)?.next_open ?? null,
      next_session: observed.get(row.ticker)?.next_session ?? null,
      close_pnl: row.hypothetical.pnl, held_overnight: held,
      overnight_status: overnight.status,
      overnight_extra: held ? overnight.incrementalPnl : 0,
      final_pnl: finalReturn === null ? null : finalReturn * 10000,
      display_total_pnl: exit.pnl === null ? null : row.hypothetical.pnl + (held ? overnight.incrementalPnl ?? 0 : 0),
      final_return: finalReturn, included_in_headline: exit.pnl !== null,
      news_flag: row.news_flag, news_checked: row.news_checked,
    });
  }
}
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
function stats(selected) {
  const priced = selected.filter(r => r.final_pnl !== null);
  const pnl = sum(priced.map(r => r.final_pnl));
  return { picks: selected.length, priced: priced.length, pnl, mean_pnl: pnl / priced.length,
    mean_return: pnl / priced.length / 10000,
    wins: priced.filter(r => r.final_pnl > 0).length, losses: priced.filter(r => r.final_pnl < 0).length };
}
const included = rows.filter(r => r.included_in_headline);
const completeDays = days.filter(r => r.complete);
const summary = {
  headline: { ...history, average_daily_capital: capital, period_gain: returnOnAverageDailyCapital(history.pnl, capital),
    positions: included.length, average_per_position: history.pnl / included.length,
    average_per_day: history.pnl / history.matchedDays },
  footer: stats(rows), footer_held: stats(rows.filter(r => r.held_overnight)),
  completed: stats(included),
  completed_close_only_pnl: sum(completeDays.map(r => r.close_pnl)),
  completed_overnight_increment: sum(completeDays.map(r => r.extraPnl)),
  per_rank: [1, 2].map(rank => ({ rank, ...stats(included.filter(r => r.rank === rank)) })),
  costs: [0, 11, 26].map(bps => {
    // Same cash budget, buy quantity reduced by modeled entry friction; exit friction on proceeds.
    const cost = bps / 10000;
    const pnl = sum(included.map(r => (10000 + r.display_total_pnl) * (1 - cost) / (1 + cost) - 10000));
    return { bps_per_side: bps, pnl, average_per_position: pnl / included.length,
      at_2500_per_position: pnl / included.length / 4, period_gain: pnl / capital };
  }),
  latest_day: days[0], latest_rows: rows.filter(r => r.date === days[0].date),
  missing_or_mismatched: rows.filter(r => !['observed', 'awaiting_next_open'].includes(r.overnight_status)),
  replay_days_excluded: book.days.filter(d => d.scan_id).map(d => ({ date: d.as_of, scan_id: d.scan_id })),
};
write('dashboard_rows.json', rows);
write('dashboard_days.json', days);
write('dashboard_summary.json', summary);
console.log(JSON.stringify(summary, null, 2));
