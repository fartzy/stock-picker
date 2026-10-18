import type { FirstTouch } from "./api";
import type { OvernightExitRule } from "./whatIf";

export interface HoldBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface HoldExit {
  returnPct: number;
  reason: "target" | "stop" | "deadline";
  day: number;
}

export type { FirstTouch };

/** Looks up the minute-bar answer for a session where both levels printed. */
export type TieBreak = (session: string, target: number, stop: number) => FirstTouch | undefined;

/** Session where the daily high reached `target` and the low reached `stop`, after an
 * open between them: the only case daily bars cannot order. */
export function isTie(bar: HoldBar, target: number, stop: number): boolean {
  return bar.open > stop && bar.open < target && bar.high >= target && bar.low <= stop;
}

export function holdLevels(close: number, gainPct: number, dropPct: number): { target: number; stop: number } {
  return { target: close * (1 + gainPct / 100), stop: close * (1 - dropPct / 100) };
}

/** Exit for shares still held at `close`. Each later session checks its open
 * first: an open at or past either level fills at the open, so a gap through the
 * target sells there and a gap through the stop sells at the open, not the stop.
 * When both levels print inside one session, `tieBreak` (minute bars) decides; if
 * it has no answer, or both printed in the same minute, the stop wins. If neither
 * prints, sell at the last session's close; until that session exists the pick
 * stays pending.
 */
export function holdExit(
  close: number,
  sessions: readonly HoldBar[],
  gainPct: number,
  dropPct: number,
  days: number,
  tieBreak?: TieBreak,
): HoldExit | null {
  if (!close) return null;
  const window = sessions.slice(0, days);
  const { target, stop } = holdLevels(close, gainPct, dropPct);
  const exit = (price: number, reason: HoldExit["reason"], index: number): HoldExit => (
    { returnPct: price / close - 1, reason, day: index + 1 }
  );
  for (const [index, bar] of window.entries()) {
    if (bar.open <= stop) return exit(bar.open, "stop", index);
    if (bar.open >= target) return exit(bar.open, "target", index);
    if (isTie(bar, target, stop) && tieBreak?.(bar.date, target, stop) === "target") {
      return exit(target, "target", index);
    }
    if (bar.low <= stop) return exit(stop, "stop", index);
    if (bar.high >= target) return exit(target, "target", index);
  }
  // No exit yet; the deadline close needs every session in the wait.
  if (window.length < days) return null;
  return exit(window[window.length - 1].close, "deadline", window.length - 1);
}

/** Key for one tie lookup. Prices are rounded so browser and API floats agree. */
export function tieKey(ticker: string, session: string, target: number, stop: number): string {
  return `${ticker}:${session}:${target.toFixed(4)}:${stop.toFixed(4)}`;
}

/** Every (ticker, session, target, stop) where daily bars cannot order the levels. */
export function tieCases(
  rows: readonly HoldRow[],
  paths: ReadonlyMap<string, readonly HoldBar[]>,
  gainPct: number,
  dropPct: number,
  days: number,
): Array<{ ticker: string; session: string; target: number; stop: number }> {
  const cases = [];
  for (const row of rows) {
    if (row.hypothetical.status !== "scored" && row.hypothetical.status !== "held") continue;
    if (!row.close_price) continue;
    const { target, stop } = holdLevels(row.close_price, gainPct, dropPct);
    for (const bar of (paths.get(row.ticker) ?? []).slice(0, days)) {
      if (bar.open <= stop || bar.open >= target) break;
      if (isTie(bar, target, stop)) {
        cases.push({ ticker: row.ticker, session: bar.date, target, stop });
        break;
      }
      if (bar.low <= stop || bar.high >= target) break;
    }
  }
  return cases;
}

export interface HoldSummary {
  pnl: number;
  avg: number;
  n: number;
  pending: number;
}

export function addHoldSummaries(parts: Array<HoldSummary | null>): HoldSummary | null {
  const present = parts.filter((part): part is HoldSummary => part !== null);
  if (present.length === 0) return null;
  const n = present.reduce((total, part) => total + part.n, 0);
  const pending = present.reduce((total, part) => total + part.pending, 0);
  const pnl = present.reduce((total, part) => total + part.pnl, 0);
  const weighted = present.reduce((total, part) => total + part.avg * part.n, 0);
  return {
    pnl: Math.round(pnl * 100) / 100,
    avg: n ? weighted / n : 0,
    n,
    pending,
  };
}

interface HoldRow {
  ticker: string;
  open_price: number | null;
  close_price: number | null;
  hypothetical: { status: string; endingValue: number | null };
}

/** Whole-trade result: buy at the day's open, hold through the close, then
 * exit by `holdExit`. Same scored/held picks as the close-only baseline, so
 * the two totals compare directly.
 */
export function summarizeHold(
  rows: readonly HoldRow[],
  paths: ReadonlyMap<string, readonly HoldBar[]>,
  gainPct: number,
  dropPct: number,
  days: number,
  dollarsPerTrade: number,
  tieBreaks?: ReadonlyMap<string, FirstTouch>,
): HoldSummary | null {
  let pnlCents = 0;
  let total = 0;
  let n = 0;
  let pending = 0;
  for (const row of rows) {
    if (row.hypothetical.status !== "scored" && row.hypothetical.status !== "held") continue;
    const { open_price: open, close_price: close } = row;
    const ending = row.hypothetical.endingValue;
    const exit = open && close && ending !== null
      ? holdExit(close, paths.get(row.ticker) ?? [], gainPct, dropPct, days,
        tieBreaks && ((session, target, stop) => tieBreaks.get(tieKey(row.ticker, session, target, stop))))
      : null;
    if (!exit || !open || !close || ending === null) {
      pending += 1;
      continue;
    }
    const value = ending + dollarsPerTrade / open * close * exit.returnPct;
    const pnl = Math.round((value - dollarsPerTrade) * 100);
    pnlCents += pnl;
    total += pnl / 100 / dollarsPerTrade;
    n += 1;
  }
  if (n === 0 && pending === 0) return null;
  return { pnl: pnlCents / 100, avg: n ? total / n : 0, n, pending };
}

export type AfterCloseMode = "close" | "next-open" | "target";

export interface AfterCloseChoice {
  mode: AfterCloseMode;
  gainPct: number;
  dropPct: number;
  days: number;
  rule: OvernightExitRule;
  portion: number;
}

export interface ResolvedAfterClose {
  nextOpen: { rule: OvernightExitRule; portion: number } | null;
  hold: { gainPct: number; dropPct: number; days: number } | null;
}

/** A +0% target with a one-day wait sells at the next morning's open: the open
 * is the first price after the close and already meets a 0% target, so the stop
 * never gets a turn. Route it through the Next morning open calculation (all
 * picks, 100% held) so both settings share one formula, one price source, and
 * one set of priced picks.
 */
export function resolveAfterClose(choice: AfterCloseChoice): ResolvedAfterClose {
  if (choice.mode === "next-open") return { nextOpen: { rule: choice.rule, portion: choice.portion }, hold: null };
  if (choice.mode !== "target") return { nextOpen: null, hold: null };
  if (choice.gainPct === 0 && choice.days === 1) return { nextOpen: { rule: "all", portion: 100 }, hold: null };
  return { nextOpen: null, hold: { gainPct: choice.gainPct, dropPct: choice.dropPct, days: choice.days } };
}
