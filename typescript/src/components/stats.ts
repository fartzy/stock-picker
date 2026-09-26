/** Shared stats UI: tables, strips, signed numbers, named columns. */

export { Diff, SignedPct } from "./Diff";
export { StatStrip, type StatItem } from "./StatStrip";
export {
  StatBody,
  StatDetail,
  StatExpand,
  StatHead,
  StatRow,
  StatTable,
  type StatAppearance,
  type StatColumn,
  type StatValues,
} from "./StatTable";
export {
  CLOSED_LOT_COLUMNS,
  HOLD_WINDOW_LABEL,
  OPEN_LOT_COLUMNS,
  PERIOD_COLUMNS,
  PeriodCol,
  col,
  lotColumnClass,
} from "./statColumns";
