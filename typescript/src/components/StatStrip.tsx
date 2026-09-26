import type { ReactNode } from "react";

/** One-off labeled facts (Open now, freshness). Repeating grids use StatTable. */
export type StatItem = {
  key: string;
  label?: string;
  value?: ReactNode;
  align?: "start" | "end";
  title?: boolean;
};

export function StatStrip({ items, className }: { items: StatItem[]; className?: string }) {
  return (
    <span className={["stat-strip", className].filter(Boolean).join(" ")}>
      {items.map((item) => (
        <span
          key={item.key}
          className={item.title ? "stat-cell is-title" : "stat-cell"}
          data-align={item.align ?? (item.title ? "start" : "end")}
        >
          {item.label ? <span className="stat-label">{item.label}</span> : null}
          <span className="stat-value">{item.value}</span>
        </span>
      ))}
    </span>
  );
}
