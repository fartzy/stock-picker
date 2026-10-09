import { fetchQuotes, type QuoteSummary, type QuotesResponse } from "./api";
import { useFetchData } from "./useFetchData";

export function quotesByTicker(quotes: QuoteSummary[] | null | undefined): Record<string, QuoteSummary> {
  const out: Record<string, QuoteSummary> = {};
  for (const quote of quotes ?? []) {
    out[quote.ticker] = quote;
  }
  return out;
}

export function useQuotes(
  tickers: string[],
  options?: { enabled?: boolean; intervalMs?: number; asOf?: string },
): { quotes: Record<string, QuoteSummary>; error: string | null; loading: boolean } {
  const enabled = options?.enabled ?? true;
  const unique = [...new Set(tickers.filter(Boolean))].sort();
  const key = enabled ? unique.join(",") : "";
  const asOf = options?.asOf;
  const { data, error, loading } = useFetchData<QuotesResponse>(
    () => (enabled && unique.length > 0 ? fetchQuotes(unique, asOf) : Promise.resolve({ quotes: [] })),
    {
      deps: [key, enabled, asOf],
      intervalMs: enabled && unique.length > 0 ? options?.intervalMs : undefined,
      trackLoading: true,
    },
  );
  return { quotes: quotesByTicker(data?.quotes), error, loading: enabled && unique.length > 0 && loading };
}
