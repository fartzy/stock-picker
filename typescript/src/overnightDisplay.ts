export type OvernightDirection = "up" | "down" | "flat";

export function overnightQuoteStatus(fresh: boolean, withinCashSession: boolean): string {
  if (fresh) return "Fresh";
  return withinCashSession ? "Stale — refresh before use" : "Outside regular hours — enter a price manually";
}

// The headline shows cents and hundredths of a percent. A smaller move should
// not appear as "Up +0.00%" beside an unchanged-looking price.
export function overnightDirection(predictedGap: number, differencePerShare: number): OvernightDirection {
  if (Math.abs(predictedGap) < 0.00005 && Math.abs(differencePerShare) < 0.005) return "flat";
  return predictedGap > 0 ? "up" : predictedGap < 0 ? "down" : "flat";
}

export function overnightGapText(predictedGap: number, direction: OvernightDirection): string {
  if (direction === "flat") return predictedGap === 0 ? "0.00%" : "≈0%";
  const magnitude = Math.abs(predictedGap * 100);
  if (magnitude < 0.005) return "<0.01%";
  return `${predictedGap > 0 ? "+" : "-"}${magnitude.toFixed(2)}%`;
}

export function overnightDollarText(differencePerShare: number): string {
  const magnitude = Math.abs(differencePerShare);
  if (magnitude > 0 && magnitude < 0.005) return "<$0.01";
  const dollars = magnitude.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${differencePerShare > 0 ? "+" : differencePerShare < 0 ? "-" : ""}$${dollars}`;
}
