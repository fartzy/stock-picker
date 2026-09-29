import type { PaperBookDay, PaperBookResponse, PaperListStats, PaperPickRow } from "./api";

export const DEFAULT_TRADE_SIZE = 10_000;
export const TRADE_SIZE_OPTIONS = [5_000, 10_000] as const;
export const LOOKBACK_OPTIONS = [
  { days: null, label: "All history" },
  { days: 14, label: "Past 2 weeks" },
  { days: 21, label: "Past 3 weeks" },
] as const;
export type LookbackDays = (typeof LOOKBACK_OPTIONS)[number]["days"];
export type PickKind = "fit" | "rank" | "both";
export type TradeSizes = Readonly<Record<Exclude<PickKind, "both">, number>>;

export interface PickOutcome {
  status: "scored" | "skipped" | "pending";
  pnl: number | null;
  endingValue: number | null;
}

export interface SimulatedPick extends PaperPickRow {
  hypothetical: PickOutcome;
}

export interface MoneySummary {
  pnl: number | null;
  invested: number;
  endingValue: number | null;
  completed: number;
  pending: number;
}

export interface SimulatedDay extends Omit<PaperBookDay, "fit" | "rank"> {
  fit: SimulatedPick[];
  rank: SimulatedPick[];
  fit_money: MoneySummary;
  rank_money: MoneySummary;
}

export interface SimulatedBook extends Omit<PaperBookResponse, "days"> {
  days: SimulatedDay[];
  fit_money: MoneySummary;
  rank_money: MoneySummary;
  from: string | null;
  through: string;
}

interface SimulationOptions {
  kind: PickKind;
  fitTopK?: number;
  rankTopK?: number;
  tradeSizes: TradeSizes;
  lookbackDays: LookbackDays;
  asOf: string;
}

/** Calendar windows follow the trading dashboard's timezone, not the browser's. */
export function simulationDay(now = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Chicago" }).format(now);
}

const cents = (value: number) => Math.round(value * 100);
const hasReturn = (row: PaperPickRow) => row.session_return !== null && Number.isFinite(row.session_return);

/** Fixed notional per pick; fractional-share equivalent, before fees/slippage.
 * Round each pick once so displayed line items reconcile with every total.
 */
export function simulatePick(row: PaperPickRow, dollarsPerTrade: number): PickOutcome {
  if (!Number.isFinite(dollarsPerTrade) || dollarsPerTrade <= 0) {
    throw new RangeError("Dollars per trade must be positive and finite");
  }
  if (row.news_blocks) return { status: "skipped", pnl: null, endingValue: null };
  if (!hasReturn(row)) return { status: "pending", pnl: null, endingValue: null };
  const pnlCents = cents(dollarsPerTrade * row.session_return!);
  return { status: "scored", pnl: pnlCents / 100, endingValue: (cents(dollarsPerTrade) + pnlCents) / 100 };
}

function moneySummary(rows: SimulatedPick[], dollarsPerTrade: number): MoneySummary {
  const completed = rows.filter((row) => row.hypothetical.status === "scored");
  const pnlCents = completed.reduce((sum, row) => sum + cents(row.hypothetical.pnl!), 0);
  const investedCents = completed.length * cents(dollarsPerTrade);
  return {
    pnl: completed.length ? pnlCents / 100 : null,
    invested: investedCents / 100,
    endingValue: completed.length ? (investedCents + pnlCents) / 100 : null,
    completed: completed.length,
    pending: rows.filter((row) => row.hypothetical.status === "pending").length,
  };
}

function listStats(rows: PaperPickRow[]): PaperListStats {
  const kept = rows.filter((row) => !row.news_blocks && hasReturn(row));
  const returns = kept.map((row) => row.session_return!);
  const wins = returns.filter((value) => value > 0).length;
  const losses = returns.filter((value) => value < 0).length;
  const avg = returns.length ? returns.reduce((sum, value) => sum + value, 0) / returns.length : null;
  return {
    n: rows.length, n_scored: returns.length, wins, losses,
    flats: returns.length - wins - losses,
    hit_rate: returns.length ? wins / returns.length : null,
    avg, avg_ex_news: avg, n_avoid: rows.filter((row) => row.news_blocks).length,
  };
}

function compound(avgs: number[]): number | null {
  return avgs.length ? avgs.reduce((wealth, avg) => wealth * (1 + avg), 1) - 1 : null;
}

/** One pipeline for filtering, sizing, day results and overall results.
 * Top-K is applied before the news filter: skipped picks are not backfilled.
 * Replay rows can be inspected but never double-count a live session in totals.
 * Dollar profits are added, not compounded or multiplied by the compound metric.
 */
export function simulateBook(data: PaperBookResponse, options: SimulationOptions): SimulatedBook {
  const { kind, fitTopK, rankTopK, tradeSizes, lookbackDays, asOf } = options;
  const start = new Date(`${asOf}T12:00:00Z`);
  if (lookbackDays !== null) start.setUTCDate(start.getUTCDate() - lookbackDays + 1);
  const from = lookbackDays === null ? null : start.toISOString().slice(0, 10);
  const rowsFor = (rows: PaperPickRow[], dollarsPerTrade: number, topK?: number): SimulatedPick[] => rows
    .filter((row) => topK === undefined || row.rank <= topK)
    .map((row) => ({ ...row, hypothetical: simulatePick(row, dollarsPerTrade) }));
  const days = data.days
    .filter((day) => day.as_of <= asOf && (from === null || day.as_of >= from))
    .map((day): SimulatedDay => {
      const fit = kind === "rank" ? [] : rowsFor(day.fit, tradeSizes.fit, fitTopK);
      const rank = kind === "fit" ? [] : rowsFor(day.rank, tradeSizes.rank, rankTopK);
      const fitStats = listStats(fit), rankStats = listStats(rank);
      return {
        ...day, fit, rank, fit_avg: fitStats.avg, rank_avg: rankStats.avg,
        fit_stats: fitStats, rank_stats: rankStats,
        fit_money: moneySummary(fit, tradeSizes.fit), rank_money: moneySummary(rank, tradeSizes.rank),
      };
    });
  const liveDays = days.filter((day) => !day.scan_id);
  const fit = liveDays.flatMap((day) => day.fit), rank = liveDays.flatMap((day) => day.rank);
  const fitAvgs = liveDays.map((day) => day.fit_avg).filter((value): value is number => value !== null);
  const rankAvgs = liveDays.map((day) => day.rank_avg).filter((value): value is number => value !== null);
  return {
    ...data, days, from, through: asOf, kind,
    fit_compound: compound(fitAvgs), rank_compound: compound(rankAvgs),
    fit_days: fitAvgs.length, rank_days: rankAvgs.length,
    fit_stats: listStats(fit), rank_stats: listStats(rank),
    fit_money: moneySummary(fit, tradeSizes.fit), rank_money: moneySummary(rank, tradeSizes.rank),
    n_picks: fit.length + rank.length,
  };
}
