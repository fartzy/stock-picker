import type { Position, Trade } from "./api";

export interface TickerSuggestion {
  ticker: string;
  detail: string;
}

/** Open lots first, then distinct symbols in most-recent-trade order. */
export function tradedTickerSuggestions(
  positions: readonly Position[],
  trades: readonly Trade[],
): TickerSuggestion[] {
  const openShares = new Map<string, number>();
  for (const position of positions) {
    if (position.closed || position.shares <= 0) continue;
    const ticker = position.ticker.trim().toUpperCase();
    if (!ticker) continue;
    openShares.set(ticker, (openShares.get(ticker) ?? 0) + position.shares);
  }

  const suggestions: TickerSuggestion[] = [...openShares]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([ticker, shares]) => ({ ticker, detail: `Open · ${shares.toLocaleString()} shares` }));
  const seen = new Set(openShares.keys());
  const timestamp = (trade: Trade) => Date.parse(trade.executed_at) || 0;
  for (const trade of [...trades].sort((a, b) => timestamp(b) - timestamp(a))) {
    const ticker = trade.ticker.trim().toUpperCase();
    if (!ticker || seen.has(ticker)) continue;
    suggestions.push({ ticker, detail: "Recent trade" });
    seen.add(ticker);
  }
  return suggestions;
}
