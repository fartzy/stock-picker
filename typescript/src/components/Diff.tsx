import { formatUsd } from "../format";

export function Diff({ value, pct }: { value: number; pct: number | null }) {
  const isUp = value >= 0;
  return (
    <span className={isUp ? "quote-diff-up" : "quote-diff-down"}>
      {isUp ? "▲" : "▼"} {formatUsd(Math.abs(value))}
      {pct !== null ? ` (${(pct * 100).toFixed(2)}%)` : ""}
    </span>
  );
}
