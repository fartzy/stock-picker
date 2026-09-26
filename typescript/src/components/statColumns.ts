import type { ReactNode } from "react";
import type { StatColumn } from "./StatTable";

/** Session window used on period rows and the closed-lot table. */
export const HOLD_WINDOW_LABEL = "8:40–2:55 CT";

/** Build one StatTable column. Default align is end unless `title`. */
export function col(
  key: string,
  label?: ReactNode,
  extras: Omit<StatColumn, "key" | "label"> = {},
): StatColumn {
  const title = extras.title ?? false;
  return {
    key,
    label,
    title,
    align: extras.align ?? (title ? "start" : "end"),
    hug: extras.hug,
  };
}

/**
 * Period / week / day rows share these keys.
 * Add a column here, then fill the same key in periodValues / dayValues.
 */
export const PeriodCol = {
  title: col("title", undefined, { title: true, hug: true }),
  days: col("days", undefined, { align: "start", hug: true }),
  typical: col("typical", "Typical"),
  pnl: col("pnl", "P&L"),
  intraday: col("intraday", "Intraday S&P"),
  bh: col("bh", "Buy & hold S&P"),
  hold: col("hold", HOLD_WINDOW_LABEL),
} as const;

export const PERIOD_COLUMNS: StatColumn[] = Object.values(PeriodCol);

export const OPEN_LOT_COLUMNS = [
  "Ticker",
  "Shares",
  "Bought",
  "Buy Price",
  "Last",
  "P&L",
  "Invested",
] as const;

export const CLOSED_LOT_COLUMNS = [
  "Ticker",
  "Shares",
  "Bought",
  "Buy Price",
  "Sold",
  "Sell Price",
  "P&L",
  "Close",
  HOLD_WINDOW_LABEL,
  "Invested",
] as const;

export function lotColumnClass(column: string): string | undefined {
  return column === "Ticker" || column === "Bought" || column === "Sold" ? undefined : "trade-num";
}
