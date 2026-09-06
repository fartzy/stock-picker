export interface CatalogResponse {
  catalog: Record<string, string[]>;
  descriptions: Record<string, string>;
  formulas: Record<string, string>;
}

export interface ImportanceResponse {
  importance: Record<string, number>;
  // Per-model-type breakdown (e.g. lightgbm/random_forest/logistic_regression),
  // each its own feature -> percent-of-total map. logistic_regression is fit
  // as a standalone diagnostic model, not an ensemble member -- it never
  // contributes to `importance` above, so this is the only place its
  // coefficient-based importance is visible.
  by_model_type?: Record<string, Record<string, number>>;
}

export interface EnsembleMemberInfo {
  model_type: string;
  weight: number;
  feature_count: number;
}

export interface ModelInfoResponse {
  models: EnsembleMemberInfo[];
}

export interface ModelTypeInfo {
  model_type: string;
  display_name: string;
  category: string;
  package: string;
  package_version: string;
  source_file: string;
  source_line: number;
  // null if the origin remote couldn't be resolved -- fall back to plain
  // source_file:source_line text rather than a link.
  github_url: string | null;
  description: string;
}

export interface ModelTypesResponse {
  model_types: ModelTypeInfo[];
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

export interface FeatureValuesResponse {
  ticker: string;
  columns: string[];
  // Each row: { date: "...", <column>: value | null, ... }.
  rows: Record<string, number | string | null>[];
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

export interface FeatureSelectionResponse {
  // null = no explicit selection -- every feature, subject to pruning only.
  included_features: string[] | null;
}

export interface ModelChoice {
  model_type: string;
  weight: number;
}

export interface ModelSelectionResponse {
  // null = no explicit choice -- the backend's own default composition.
  model_choices: ModelChoice[] | null;
  // Which model types the composable picker offers. logistic_regression is
  // deliberately absent -- it's fit as a standalone diagnostic, never an
  // ensemble member (see ImportanceResponse's by_model_type comment).
  available_model_types: string[];
}

export type TrainingStatus = "idle" | "running" | "completed" | "failed";

export interface FoldMetrics {
  mae: number;
  directional_accuracy: number;
  n_test_rows: number;
}

export interface ThresholdSweepRow {
  threshold: number;
  n_trades: number;
  hit_rate: number;
  total_return: number;
  avg_return: number;
  return_std: number;
}

export interface TrainingResult {
  fold_metrics: FoldMetrics[];
  holdout_metrics: FoldMetrics | null;
  threshold_sweep: ThresholdSweepRow[] | null;
}

export interface TrainingStatusResponse {
  status: TrainingStatus;
  started_at: string | null;
  completed_at: string | null;
  result: TrainingResult | null;
  error: string | null;
}

export interface RunModelSpec {
  model_type: string;
  weight: number;
  params: Record<string, unknown> | null;
}

export interface TrainingRunRecord {
  run_id: string;
  status: "completed" | "failed";
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  git_commit: string | null;
  // null on a run that failed before this provenance was known.
  train_tickers: string[] | null;
  holdout_tickers: string[] | null;
  date_range: [string, string] | null;
  resolved_features: string[] | null;
  model_specs: RunModelSpec[] | null;
  fold_metrics: FoldMetrics[] | null;
  holdout_metrics: FoldMetrics | null;
  threshold_sweep: ThresholdSweepRow[] | null;
  error: string | null;
}

export interface TrainingRunsResponse {
  // Newest first.
  runs: TrainingRunRecord[];
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
export const fetchModelInfo = () => getJson<ModelInfoResponse>("/api/model-info");
export const fetchModelTypes = () => getJson<ModelTypesResponse>("/api/model-types");
export const fetchCoverage = () => getJson<CoverageResponse>("/api/coverage");
export const fetchPriceHistory = (ticker: string, interval: "daily" | "hourly") =>
  getJson<PriceHistoryResponse>(`/api/prices/${encodeURIComponent(ticker)}?interval=${interval}`);
export const fetchFeatureValues = (ticker: string) =>
  getJson<FeatureValuesResponse>(`/api/features/${encodeURIComponent(ticker)}`);
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
export const fetchFeatureSelection = () =>
  getJson<FeatureSelectionResponse>("/api/feature-selection");
export const setFeatureSelection = (includedFeatures: string[]) =>
  mutate<FeatureSelectionResponse>("POST", "/api/feature-selection", {
    included_features: includedFeatures,
  });
export const clearFeatureSelection = () =>
  mutate<FeatureSelectionResponse>("DELETE", "/api/feature-selection");
export const fetchModelSelection = () => getJson<ModelSelectionResponse>("/api/model-selection");
export const setModelSelection = (modelChoices: ModelChoice[]) =>
  mutate<ModelSelectionResponse>("POST", "/api/model-selection", { model_choices: modelChoices });
export const clearModelSelection = () =>
  mutate<ModelSelectionResponse>("DELETE", "/api/model-selection");
export const fetchTrainingStatus = () => getJson<TrainingStatusResponse>("/api/training/status");
export const runTraining = () => mutate<TrainingStatusResponse>("POST", "/api/training/run");
export const fetchTrainingRuns = () => getJson<TrainingRunsResponse>("/api/training/runs");
export const fetchQuotes = (tickers: string[]) =>
  getJson<QuotesResponse>(`/api/quotes?tickers=${tickers.map(encodeURIComponent).join(",")}`);
export const createTrade = (trade: TradeCreate) => mutate<TradesResponse>("POST", "/api/trades", trade);
