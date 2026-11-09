export const JEV_QUESTION_LABELS: Record<string, string> = {
  net_positive_by_close: "Profit after costs by close",
  downside_before_close: "Touch the downside level before close",
  net_positive_overnight: "Profit after costs at next open",
  buy_today: "Close above the morning open",
  hold_overnight: "Next open above the session close",
};

export function jevPercent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "Not available" : `${(value * 100).toFixed(1)}%`;
}

export function jevProbability(value: number | null | undefined): string {
  return value == null || value < 0 || value > 1 ? "Not available" : jevPercent(value);
}

export function jevPriceBasis(basis: string): string {
  const labels: Record<string, string> = {
    observed_trade: "Observed last trade",
    user_assumption: "Your assumed price",
    session_open: "Morning session open",
    assumed_close: "Assumed close",
    session_close: "Actual session close, observed later",
  };
  return labels[basis] ?? basis;
}

export function jevTime(value: string | null | undefined): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "Time unavailable";
  return new Date(value).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", second: "2-digit" });
}
