import type { ReactNode } from "react";
import { formatUsd } from "../format";
import { Diff } from "./Diff";

export type Column<T> = {
  key: string;
  header: ReactNode;
  numeric?: boolean;
  when?: boolean;
  cell: (row: T) => ReactNode;
};

export function ColumnTitle({ label, hint }: { label: string; hint?: string }) {
  return (
    <span className="col-title">
      {label}
      {hint ? <span className="col-hint">{hint}</span> : null}
    </span>
  );
}

export function DataTable<T>({
  rows,
  columns,
  rowKey,
}: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
}) {
  const visible = columns.filter((column) => column.when !== false);
  return (
    <table className="trade-table">
      <thead>
        <tr>
          {visible.map((column) => (
            <th key={column.key} className={column.numeric ? "trade-num" : undefined}>
              {column.header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={rowKey(row)}>
            {visible.map((column) => (
              <td key={column.key} className={column.numeric ? "trade-num" : undefined}>
                {column.cell(row)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TickerCell({ ticker }: { ticker: string }) {
  return <span className="trade-ticker">{ticker}</span>;
}

export function UsdCell({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  return <>{formatUsd(value)}</>;
}

export function UsdDiffCell({
  value,
  vs,
}: {
  value: number | null | undefined;
  vs: number | null | undefined;
}) {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  return (
    <>
      {formatUsd(value)}
      {vs !== null && vs !== undefined && (
        <>
          {" "}
          <Diff value={value - vs} pct={vs ? (value - vs) / vs : null} />
        </>
      )}
    </>
  );
}

export function ScoreCell({ value, isRank }: { value: number | null | undefined; isRank: boolean }) {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  return <>{isRank ? value.toFixed(4) : `${(value * 100).toFixed(2)}%`}</>;
}

export function NewsCell({
  flag,
  blocks,
}: {
  flag?: string | null;
  blocks?: boolean;
}) {
  if (!flag) return <span className="muted">—</span>;
  return (
    <span className={blocks ? "quote-diff-down" : "quote-diff-up"}>
      {blocks ? "skip" : "still buy"} · {flag}
    </span>
  );
}
