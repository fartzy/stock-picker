import { createContext, useContext, type ReactNode } from "react";

/** One column. Look lives in CSS via `data-appearance`. */
export type StatColumn = {
  key: string;
  label?: ReactNode;
  align?: "start" | "end";
  title?: boolean;
  /** Keep this column tight; leftover width is shared by the rest. */
  hug?: boolean;
};

/** CSS hook on the table. Add a block in index.css to swap look. */
export type StatAppearance = "period" | "weeks" | "days" | (string & {});

export type StatValues = Record<string, ReactNode>;

const TableCtx = createContext<StatColumn[] | null>(null);

function useColumns(): StatColumn[] {
  const columns = useContext(TableCtx);
  if (!columns) throw new Error("StatHead/StatRow must sit inside StatTable");
  return columns;
}

function cellAlign(column: StatColumn): "start" | "end" {
  return column.align ?? (column.title ? "start" : "end");
}

function cellClass(column: StatColumn): string | undefined {
  const bits = [column.title ? "is-title" : "", column.hug ? "is-hug" : ""].filter(Boolean);
  return bits.length ? bits.join(" ") : undefined;
}

type TableProps = {
  columns: StatColumn[];
  /** CSS hook. Swap look in index.css without changing callers. */
  appearance?: StatAppearance;
  className?: string;
  children: ReactNode;
};

export function StatTable({ columns, appearance = "period", className, children }: TableProps) {
  return (
    <TableCtx.Provider value={columns}>
      <table
        className={["stat-table", className].filter(Boolean).join(" ")}
        data-appearance={appearance}
      >
        <colgroup>
          {columns.map((column) => (
            <col
              key={column.key}
              className={[column.title ? "is-title" : "", column.hug ? "is-hug" : ""]
                .filter(Boolean)
                .join(" ") || undefined}
            />
          ))}
        </colgroup>
        {children}
      </table>
    </TableCtx.Provider>
  );
}

export function StatHead() {
  const columns = useColumns();
  return (
    <thead>
      <tr>
        {columns.map((column) => (
          <th key={column.key} className={cellClass(column)} data-align={cellAlign(column)}>
            {column.label ?? ""}
          </th>
        ))}
      </tr>
    </thead>
  );
}

export function StatBody({ children }: { children: ReactNode }) {
  return <tbody>{children}</tbody>;
}

export function StatRow({
  values,
  nested = false,
}: {
  values: StatValues;
  nested?: boolean;
}) {
  const columns = useColumns();
  return (
    <tr className={nested ? "is-nested" : undefined}>
      {columns.map((column) => (
        <td key={column.key} className={cellClass(column)} data-align={cellAlign(column)}>
          {values[column.key] ?? ""}
        </td>
      ))}
    </tr>
  );
}

/** Full-width body under an expanded row (lot table, etc.). */
export function StatDetail({ children }: { children: ReactNode }) {
  const columns = useColumns();
  return (
    <tbody>
      <tr className="stat-expand-body">
        <td colSpan={columns.length}>{children}</td>
      </tr>
    </tbody>
  );
}

/** One summary row as its own `<tbody>`. Children render as sibling bodies. */
export function StatExpand({
  values,
  open,
  onToggle,
  nested = false,
  children,
}: {
  values: StatValues;
  open: boolean;
  onToggle: () => void;
  nested?: boolean;
  children?: ReactNode;
}) {
  const columns = useColumns();
  return (
    <>
      <tbody>
        <tr
          className={["is-expandable", nested ? "is-nested" : ""].filter(Boolean).join(" ")}
          aria-expanded={open}
          onClick={onToggle}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              onToggle();
            }
          }}
          tabIndex={0}
        >
          {columns.map((column, index) => (
            <td key={column.key} className={cellClass(column)} data-align={cellAlign(column)}>
              {index === 0 ? (
                <span className="stat-expand" aria-hidden="true">
                  {values[column.key] ?? ""}
                </span>
              ) : (
                (values[column.key] ?? "")
              )}
            </td>
          ))}
        </tr>
      </tbody>
      {open ? children : null}
    </>
  );
}
