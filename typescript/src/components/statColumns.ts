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
    colSpan: extras.colSpan,
    width: extras.width,
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

/** The fee track stays reserved so toggling it never shifts the other columns. */
const FEES_PERCENT = 5;
const lotWidth = (percent: number) => `${(percent * (100 - FEES_PERCENT)) / 100}%`;

/**
 * One physical layout for the summary and lot tables. Summary cells span their
 * constituent lot columns; P&L, fees and hold-to-close share identical tracks.
 * Lot widths divide the space remaining after the reserved fee column.
 */
const HISTORY_COLUMN_GROUPS = [
  {
    summary: PeriodCol.title,
    lots: [
      { key: "ticker", label: "Ticker", width: lotWidth(6) },
      { key: "shares", label: "Shares", width: lotWidth(6) },
      { key: "bought", label: "Bought", width: lotWidth(13) },
    ],
  },
  {
    summary: PeriodCol.days,
    lots: [{ key: "buy-price", label: "Buy Price", width: lotWidth(7) }],
  },
  {
    summary: PeriodCol.typical,
    lots: [
      { key: "sold", label: "Sold", width: lotWidth(10) },
      { key: "sell-price", label: "Sell Price", width: lotWidth(8) },
    ],
  },
  { summary: PeriodCol.pnl, lots: [{ key: "pnl", label: "P&L", width: lotWidth(17) }] },
  { summary: PeriodCol.fees, lots: [{ key: "fees", label: "Fees", width: `${FEES_PERCENT}%` }] },
  { summary: PeriodCol.intraday, lots: [{ key: "close", label: "Close", width: lotWidth(8) }] },
  { summary: PeriodCol.bh, lots: [{ key: "invested", label: "Invested", width: lotWidth(10) }] },
  { summary: PeriodCol.hold, lots: [{ key: "hold", label: HOLD_WINDOW_LABEL, width: lotWidth(15) }] },
];

export const CLOSED_LOT_COLUMNS = HISTORY_COLUMN_GROUPS.flatMap(({ lots }) => lots);
export const PERIOD_COLUMNS: StatColumn[] = HISTORY_COLUMN_GROUPS.map(({ summary, lots }) => ({
  ...summary,
  colSpan: lots.length,
}));

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

export function lotColumnClass(column: string): string | undefined {
  if (column === "Ticker" || column === "Bought" || column === "Sold") return undefined;
  if (column === "P&L") return "trade-num lot-pnl";
  if (column === "Fees") return "trade-num lot-fees";
  return "trade-num";
}
