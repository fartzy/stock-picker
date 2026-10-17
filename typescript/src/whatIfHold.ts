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

/** Exit for shares still held at `close`. Each later session checks its open
 * first: an open at or past either level fills at the open, so a gap through the
 * target sells there and a gap through the stop sells at the open, not the stop.
 * Inside a session the order of high and low is unknown, so the stop wins when
 * both print. If neither prints, sell at the last session's close; until that
 * session exists the pick stays pending.
 */
export function holdExit(
  close: number,
  sessions: readonly HoldBar[],
  gainPct: number,
  dropPct: number,
  days: number,
): HoldExit | null {
  if (!close) return null;
  const window = sessions.slice(0, days);
  const target = close * (1 + gainPct / 100);
  const stop = close * (1 - dropPct / 100);
  const exit = (price: number, reason: HoldExit["reason"], index: number): HoldExit => (
    { returnPct: price / close - 1, reason, day: index + 1 }
  );
  for (const [index, bar] of window.entries()) {
    if (bar.open <= stop) return exit(bar.open, "stop", index);
    if (bar.open >= target) return exit(bar.open, "target", index);
    if (bar.low <= stop) return exit(stop, "stop", index);
    if (bar.high >= target) return exit(target, "target", index);
  }
  // No exit yet; the deadline close needs every session in the wait.
  if (window.length < days) return null;
  return exit(window[window.length - 1].close, "deadline", window.length - 1);
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
      ? holdExit(close, paths.get(row.ticker) ?? [], gainPct, dropPct, days)
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
