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

export function holdExit(
  close: number,
  sessions: readonly HoldBar[],
  gainPct: number,
  dropPct: number,
  days: number,
): HoldExit | null {
  const window = sessions.slice(0, days);
  if (!close || window.length < days) return null;
  // A 0% gain with a one-day wait is the next morning's open. The open is
  // already the fill, so the day's later high and low do not replace it.
  if (gainPct === 0 && days === 1) {
    const open = window[0].open;
    if (!open) return null;
    return { returnPct: open / close - 1, reason: "target", day: 1 };
  }
  const target = close * (1 + gainPct / 100);
  const stop = close * (1 - dropPct / 100);
  for (const [index, bar] of window.entries()) {
    const stopped = bar.low <= stop;
    const filled = bar.high >= target;
    if (stopped) return { returnPct: stop / close - 1, reason: "stop", day: index + 1 };
    if (filled) return { returnPct: target / close - 1, reason: "target", day: index + 1 };
  }
  const last = window[window.length - 1];
  return { returnPct: last.close / close - 1, reason: "deadline", day: window.length };
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

export function summarizeHold(
  rows: ReadonlyArray<{ close_price: number | null; open_price?: number | null; hypothetical?: { endingValue: number | null } }>,
  paths: ReadonlyMap<string, readonly HoldBar[]>,
  gainPct: number,
  dropPct: number,
  days: number,
  dollarsPerTrade: number,
): { pnl: number; avg: number; n: number; pending: number } | null {
  let pnl = 0;
  let total = 0;
  let n = 0;
  let pending = 0;
  const nextMorning = gainPct === 0 && days === 1;
  for (const row of rows) {
    if (row.close_price === null) continue;
    const ticker = "ticker" in row ? String(row.ticker) : "";
    const sessions = paths.get(ticker) ?? [];
    if (nextMorning) {
      const nextOpen = sessions[0]?.open;
      const open = row.open_price;
      const ending = row.hypothetical?.endingValue;
      if (!nextOpen || !open || ending === null || ending === undefined) {
        pending += 1;
        continue;
      }
      // Same sale as Next morning open: keep the day's open-to-close trade,
      // then add the move from that close to the next morning's open.
      const overnight = dollarsPerTrade / open * (nextOpen - row.close_price);
      pnl += ending + overnight - dollarsPerTrade;
      total += (ending + overnight) / dollarsPerTrade - 1;
      n += 1;
      continue;
    }
    const exit = holdExit(row.close_price, sessions, gainPct, dropPct, days);
    if (!exit) {
      pending += 1;
      continue;
    }
    pnl += dollarsPerTrade * exit.returnPct;
    total += exit.returnPct;
    n += 1;
  }
  if (n === 0 && pending === 0) return null;
  return {
    pnl: Math.round(pnl * 100) / 100,
    avg: n ? total / n : 0,
    n,
    pending,
  };
}
