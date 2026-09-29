/**
 * Stats kit — Trade History, morning headings, Test run, What if.
 *
 * Add a period column: PeriodCol in statColumns.ts, then fill that key
 * in periodValues / dayValues. Swap look: data-appearance + CSS, or
 * tone on col().
 */

export { Diff, SignedPct } from "./Diff";
export { StatStrip, type StatItem } from "./StatStrip";
export {
  StatBody,
  StatColGroup,
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
  periodColumns,
} from "./statColumns";
