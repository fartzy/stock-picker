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
    tone: extras.tone,
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
  pnl: col("pnl", "P&L", { tone: "pnl" }),
  fees: col("fees", undefined, { hug: true }),
  intraday: col("intraday", "Intraday S&P", { tone: "bench" }),
  bh: col("bh", "Buy & hold S&P", { tone: "bench" }),
  hold: col("hold", HOLD_WINDOW_LABEL, { tone: "bench" }),
} as const;

export const PERIOD_COLUMNS: StatColumn[] = Object.values(PeriodCol);

export function periodColumns(feesLabel: ReactNode): StatColumn[] {
  return PERIOD_COLUMNS.map((column) =>
    column.key === "fees" ? { ...column, label: feesLabel } : column,
  );
}

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
  "Fees",
  "Close",
  HOLD_WINDOW_LABEL,
  "Invested",
] as const;

export function lotColumnClass(column: string): string | undefined {
  if (column === "Ticker" || column === "Bought" || column === "Sold") return undefined;
  if (column === "Fees") return "trade-num lot-fees";
  return "trade-num";
}
