export interface HoldBar {
  date: string;
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
  rows: ReadonlyArray<{ close_price: number | null }>,
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
  for (const row of rows) {
    if (row.close_price === null) continue;
    const ticker = "ticker" in row ? String(row.ticker) : "";
    const exit = holdExit(row.close_price, paths.get(ticker) ?? [], gainPct, dropPct, days);
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
