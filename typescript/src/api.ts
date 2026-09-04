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

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const fetchCatalog = () => getJson<CatalogResponse>("/api/catalog");
export const fetchCoverage = () => getJson<CoverageResponse>("/api/coverage");
export const fetchCorrelation = () => getJson<CorrelationResponse>("/api/correlation");
export const fetchRegistry = () => getJson<RegistryResponse>("/api/registry");
