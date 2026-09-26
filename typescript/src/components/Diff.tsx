import { formatUsd } from "../format";

function diffClass(isUp: boolean): string {
  return isUp ? "quote-diff quote-diff-up" : "quote-diff quote-diff-down";
}

/** Dollar P&L with optional percent. Mark sits in a fixed slot so rows line up. */
export function Diff({ value, pct }: { value: number; pct: number | null }) {
  const isUp = value >= 0;
  return (
    <span className={diffClass(isUp)}>
      <span className="quote-diff-mark">{isUp ? "▲" : "▼"}</span>
      <span className="quote-diff-amt">{formatUsd(Math.abs(value))}</span>
      {pct !== null ? <span className="quote-diff-pct">({(pct * 100).toFixed(2)}%)</span> : null}
    </span>
  );
}

/** Percent-only signed value (S&P columns). Same mark slot as Diff. */
export function SignedPct({ value }: { value: number }) {
  const isUp = value >= 0;
  return (
    <span className={diffClass(isUp)}>
      <span className="quote-diff-mark">{isUp ? "▲" : "▼"}</span>
      <span className="quote-diff-pct">{(Math.abs(value) * 100).toFixed(2)}%</span>
    </span>
  );
}
