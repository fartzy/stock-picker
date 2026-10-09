import { createContext, useContext, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";

/** Look lives in CSS: `data-appearance` plus `tone` classes. */
export type StatColumn = {
  key: string;
  label?: ReactNode;
  align?: "start" | "end";
  title?: boolean;
  /** Tight column; leftover width is shared by the rest. */
  hug?: boolean;
  /** `bench` = S&P / session band; `pnl` = outer box around P&L. */
  tone?: "bench" | "pnl";
  /** Physical columns covered by a summary cell in a shared layout. */
  colSpan?: number;
  width?: CSSProperties["width"];
};

export type StatAppearance = "period" | "weeks" | "days" | (string & {});
export type StatValues = Record<string, ReactNode>;

const TableCtx = createContext<StatColumn[] | null>(null);

function useColumns(): StatColumn[] {
  const columns = useContext(TableCtx);
  if (!columns) throw new Error("StatHead/StatRow must sit inside StatTable");
  return columns;
}

function cx(...bits: Array<string | false | undefined>): string | undefined {
  const joined = bits.filter(Boolean).join(" ");
  return joined || undefined;
}

function columnClass(column: StatColumn): string | undefined {
  return cx(column.title && "is-title", column.hug && "is-hug", column.tone && `stat-tone-${column.tone}`);
}

function columnAlign(column: StatColumn): "start" | "end" {
  return column.align ?? (column.title ? "start" : "end");
}

function Cells({
  values,
  as,
  first,
}: {
  values?: StatValues;
  as: "th" | "td";
  first?: "expand";
}) {
  const Tag = as;
  const header = as === "th";
  return (
    <>
      {useColumns().map((column, index) => {
        const content = header ? (column.label ?? "") : (values?.[column.key] ?? "");
        return (
          <Tag
            key={column.key}
            className={columnClass(column)}
            data-align={columnAlign(column)}
            data-col={column.key}
            colSpan={column.colSpan}
          >
            {first === "expand" && index === 0 ? (
              <span className="stat-expand" aria-hidden="true">
                {content}
              </span>
            ) : (
              content
            )}
          </Tag>
        );
      })}
    </>
  );
}

/** Reuse the same physical widths in summary and expanded detail tables. */
export function StatColGroup({ columns }: { columns: StatColumn[] }) {
  return (
    <colgroup>
      {columns.map((column) => (
        <col
          key={column.key}
          className={columnClass(column)}
          data-col={column.key}
          span={column.colSpan}
          style={{ width: column.width }}
        />
      ))}
    </colgroup>
  );
}

export function StatTable({
  columns,
  layoutColumns = columns,
  appearance = "period",
  className,
  children,
}: {
  columns: StatColumn[];
  layoutColumns?: StatColumn[];
  appearance?: StatAppearance;
  className?: string;
  children: ReactNode;
}) {
  return (
    <TableCtx.Provider value={columns}>
      <table className={cx("stat-table", className)} data-appearance={appearance}>
        <StatColGroup columns={layoutColumns} />
        {children}
      </table>
    </TableCtx.Provider>
  );
}

export function StatHead() {
  return (
    <thead>
      <tr>
        <Cells as="th" />
      </tr>
    </thead>
  );
}

export function StatBody({ children }: { children: ReactNode }) {
  return <tbody>{children}</tbody>;
}

export function StatRow({ values, nested = false }: { values: StatValues; nested?: boolean }) {
  return (
    <tr className={nested ? "is-nested" : undefined}>
      <Cells as="td" values={values} />
    </tr>
  );
}

/** Full-width body under an expanded row (lot table, etc.). */
export function StatDetail({ children }: { children: ReactNode }) {
  const columns = useColumns();
  return (
    <tbody>
      <tr className="stat-expand-body">
        <td colSpan={columns.reduce((total, column) => total + (column.colSpan ?? 1), 0)}>
          {children}
        </td>
      </tr>
    </tbody>
  );
}

function onActivate(event: KeyboardEvent, onToggle: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    onToggle();
  }
}

/** One summary row as its own `<tbody>`. Open children render as sibling bodies. */
export function StatExpand({
  values,
  open,
  onToggle,
  nested = false,
  className,
  children,
}: {
  values: StatValues;
  open: boolean;
  onToggle: () => void;
  nested?: boolean;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <>
      <tbody>
        <tr
          className={cx("is-expandable", nested && "is-nested", className)}
          aria-expanded={open}
          tabIndex={0}
          onClick={onToggle}
          onKeyDown={(event) => onActivate(event, onToggle)}
        >
          <Cells as="td" values={values} first="expand" />
        </tr>
      </tbody>
      {open ? children : null}
    </>
  );
}
