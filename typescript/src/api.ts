export interface CatalogResponse {
  catalog: Record<string, string[]>;
  descriptions: Record<string, string>;
}

export interface CoverageResponse {
  coverage: Record<string, number>;
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
}

export interface TradesResponse {
  trades: Trade[];
}

export interface Position {
  ticker: string;
  day: string;
  shares: number;
  invested: number;
  buy_time: string;
  buy_price: number;
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

export interface PrunedFeaturesResponse {
  pruned_features: string[];
}

export interface QuoteSummary {
  ticker: string;
  open: number;
  last: number;
  diff: number;
  diff_pct: number;
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
export const fetchCoverage = () => getJson<CoverageResponse>("/api/coverage");
export const fetchCorrelation = () => getJson<CorrelationResponse>("/api/correlation");
export const fetchRegistry = () => getJson<RegistryResponse>("/api/registry");
export const fetchTrades = () => getJson<TradesResponse>("/api/trades");
export const fetchPositions = () => getJson<PositionsResponse>("/api/positions");
export const fetchPrunedFeatures = () => getJson<PrunedFeaturesResponse>("/api/pruned-features");
export const pruneFeature = (feature: string) => mutate("POST", `/api/features/${encodeURIComponent(feature)}/prune`);
export const unpruneFeature = (feature: string) =>
  mutate("DELETE", `/api/features/${encodeURIComponent(feature)}/prune`);
export const fetchQuotes = (tickers: string[]) =>
  getJson<QuotesResponse>(`/api/quotes?tickers=${tickers.map(encodeURIComponent).join(",")}`);
export const createTrade = (trade: TradeCreate) => mutate<TradesResponse>("POST", "/api/trades", trade);
