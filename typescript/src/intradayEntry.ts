/** User-facing rules shared by the entry panel and its focused tests. */
export const MAX_ENTRY_TICKERS = 20;

export function parseEntryTickers(value: string): { tickers: string[]; error: string | null } {
  const tickers = [...new Set(value.trim().toUpperCase().split(/[\s,;]+/).filter(Boolean))];
  if (!tickers.length) return { tickers, error: "Enter a ticker or choose the morning Rank picks." };
  const invalid = tickers.find((ticker) => !/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker));
  if (invalid) return { tickers, error: `Check the ticker “${invalid}”.` };
  if (tickers.length > MAX_ENTRY_TICKERS) {
    return { tickers, error: `Choose up to ${MAX_ENTRY_TICKERS} tickers per refresh.` };
  }
  return { tickers, error: null };
}

export function entryProbability(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value < 0 || value > 1) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

export function entryReturn(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

/** Compare the primary Rank #1 cohort, never substitute pooled performance. */
export function entryEvidenceMessage(primary: { brier: number | null; baseline_brier: number | null } | null | undefined): string {
  if (primary?.brier == null || primary.baseline_brier == null
      || !Number.isFinite(primary.brier) || !Number.isFinite(primary.baseline_brier)) {
    return "Rank #1 baseline comparison is not available yet.";
  }
  return primary.brier < primary.baseline_brier
    ? "Rank #1 odds beat the time-of-day baseline in this held-out test."
    : "Rank #1 odds did not beat the time-of-day baseline in held-out testing.";
}

export function entryPrice(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value <= 0) return "—";
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

export function entryClock(value: string | null | undefined): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "—";
  return new Date(value).toLocaleTimeString("en-US", { timeZone: "America/Chicago", hour: "numeric", minute: "2-digit", second: "2-digit" });
}

/** A zero probability is real evidence, never an empty/unavailable value. */
export function entryScoreAvailable(row: {
  status: string;
  probability_profit: number | null;
  expected_return: number | null;
  probability_loss_2pct: number | null;
}): boolean {
  return row.status === "ok"
    && entryProbability(row.probability_profit) !== "—"
    && entryProbability(row.probability_loss_2pct) !== "—"
    && entryReturn(row.expected_return) !== "—";
}
