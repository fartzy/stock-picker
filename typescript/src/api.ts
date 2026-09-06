export interface CatalogResponse {
  catalog: Record<string, string[]>;
  descriptions: Record<string, string>;
  formulas: Record<string, string>;
}

export interface ImportanceResponse {
  importance: Record<string, number>;
  // Per-model-type breakdown (e.g. lightgbm/random_forest/logistic_regression),
  // each its own feature -> percent-of-total map. logistic_regression is a
  // diagnostic-only ensemble member (weight 0) -- it never contributes to
  // `importance` above, so this is the only place its coefficient-based
  // importance is visible.
  by_model_type?: Record<string, Record<string, number>>;
}

export interface CoverageResponse {
  coverage: Record<string, number>;
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PriceHistoryResponse {
  ticker: string;
  interval: "daily" | "hourly";
  prices: PricePoint[];
}

export interface CorrelationPair {
  a: string;
  b: string;
  correlation: number;
}

export interface CorrelationResponse {
  columns: string[];
  matrix: (number | null)[][];
  top_pairs: CorrelationPair[];
}

export interface Trade {
  ticker: string;
  side: "buy" | "sell";
  shares: number;
  price: number;
  notional: number;
  executed_at: string;
  // null for a "buy" row -- only a closing "sell" has a realized P&L.
  realized_pnl: number | null;
}

export interface TradesResponse {
  trades: Trade[];
}

export interface Position {
  ticker: string;
  day: string;
  shares: number;
  invested: number;
  buy_time: string | null;
  buy_price: number | null;
  day_open: number | null;
  prev_close: number | null;
  gap: number | null;
  gap_pct: number | null;
  sell_time: string | null;
  sell_price: number | null;
  current_price: number | null;
  closed: boolean;
  pnl: number | null;
}

export interface PositionsResponse {
  positions: Position[];
}

export interface FeatureView {
  name: string;
  entities: string[];
  features: string[];
  source: string;
  ttl_days: number;
  tags: Record<string, string>;
  owner: string;
}

export interface FeatureService {
  name: string;
  feature_views: string[];
  description: string;
}

export interface RegistryResponse {
  entities: { name: string; description: string }[];
  feature_views: FeatureView[];
  feature_services: FeatureService[];
}

export interface PrunedFeatureEntry {
  feature: string;
  reason: string;
  pruned_at: string;
}

export interface PrunedFeaturesResponse {
  pruned_features: string[];
  archive: PrunedFeatureEntry[];
}

export interface QuoteSummary {
  ticker: string;
  open: number;
  last: number;
  diff: number;
  diff_pct: number;
  prev_close: number | null;
  gap: number | null;
  gap_pct: number | null;
}

export interface QuotesResponse {
  quotes: QuoteSummary[];
}

export interface TradeCreate {
  ticker: string;
  side: "buy" | "sell";
  shares: number;
  price: number;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function mutate<T = void>(method: "POST" | "DELETE", path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    throw new Error(`${method} ${path} failed: ${response.status}`);
  }
  return response.status === 204 ? (undefined as T) : (response.json() as Promise<T>);
}

export const fetchCatalog = () => getJson<CatalogResponse>("/api/catalog");
export const fetchFeatureImportance = () => getJson<ImportanceResponse>("/api/feature-importance");
export const fetchCoverage = () => getJson<CoverageResponse>("/api/coverage");
export const fetchPriceHistory = (ticker: string, interval: "daily" | "hourly") =>
  getJson<PriceHistoryResponse>(`/api/prices/${encodeURIComponent(ticker)}?interval=${interval}`);
export const fetchCorrelation = () => getJson<CorrelationResponse>("/api/correlation");
export const fetchRegistry = () => getJson<RegistryResponse>("/api/registry");
export const fetchTrades = () => getJson<TradesResponse>("/api/trades");
export const fetchPositions = () => getJson<PositionsResponse>("/api/positions");
export const fetchPrunedFeatures = () => getJson<PrunedFeaturesResponse>("/api/pruned-features");
export const pruneFeature = (feature: string, reason?: string) =>
  mutate<PrunedFeaturesResponse>(
    "POST",
    `/api/features/${encodeURIComponent(feature)}/prune`,
    reason ? { reason } : undefined,
  );
export const unpruneFeature = (feature: string) =>
  mutate<PrunedFeaturesResponse>("DELETE", `/api/features/${encodeURIComponent(feature)}/prune`);
export const fetchQuotes = (tickers: string[]) =>
  getJson<QuotesResponse>(`/api/quotes?tickers=${tickers.map(encodeURIComponent).join(",")}`);
export const createTrade = (trade: TradeCreate) => mutate<TradesResponse>("POST", "/api/trades", trade);
