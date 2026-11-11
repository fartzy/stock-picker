import { requestJson } from "./requests.js";

export interface JevForecast {
  id: string;
  question_set_version: "morning_v1" | "entry_v1" | "overnight_v1";
  ticker: string;
  session: string;
  kind: "rank" | "fit";
  requested_at: string;
  completed_at: string;
  model: string | null;
  issue: string | null;
  target: {
    entry_price: number | null;
    entry_basis: string;
    entry_observed_at: string | null;
    exit_at: string;
    next_session?: string;
    round_trip_cost_per_share: number;
    downside_fraction?: number;
    downside_price?: number;
    barrier_start_at?: string;
  };
  answers: Record<string, number | null>;
  state?: {
    news?: { articles?: Array<{ headline: string | null; summary: string | null; source: string | null; published_at: string | null }> };
  };
  coverage: {
    news_status: string;
    news_checked_at: string | null;
    news_sources: string[];
    news_issues: string[];
    feature_snapshot_date: string | null;
    issues: string[];
  };
}

export interface JevAssessmentRequest {
  ticker: string;
  kind: "rank" | "fit";
  mode: "entry" | "overnight";
  entry_price?: number;
  round_trip_cost_per_share: number;
  downside_fraction: number;
}

export interface JevReturns {
  n: number;
  mean_net_return: number | null;
  win_rate: number | null;
  losses: number;
  worst_net_return: number | null;
}

export interface JevReliabilityBin {
  lower: number;
  upper: number;
  n: number;
  mean_probability: number | null;
  event_rate: number | null;
  mean_net_return: number | null;
  loss_rate: number | null;
}

export interface JevTargetMetrics {
  question: string;
  n: number;
  brier: number | null;
  mean_probability: number | null;
  event_rate: number | null;
  reliability: JevReliabilityBin[];
  prior_day_baseline_brier: number | null;
  matched_jev_brier: number | null;
  baseline_n: number;
  unresolved: number;
}

export interface JevCohort {
  question_set_version: string;
  model: string;
  kind: string;
  entry_basis: string;
  records: number;
  round_trip_cost_per_share: number | null;
  downside_fraction: number | null;
  completed: number;
  sessions: number;
  returns: JevReturns;
  targets: JevTargetMetrics[];
  comparisons: Array<JevReturns & { name: string }>;
}

export interface JevOutcomeRow {
  forecast_id: string;
  ticker: string;
  session: string;
  kind: string;
  question_set_version: string;
  model: string | null;
  requested_at: string;
  primary: boolean;
  traded: boolean;
  status: string;
  reason: string | null;
  net_return: number | null;
  labels: Record<string, boolean | null>;
}

export interface JevEvaluation {
  generated_at: string;
  forecast_count: number;
  primary_count: number;
  duplicate_count: number;
  legacy_count: number;
  failed_count: number;
  counts: { completed: number; pending: number; unavailable: number };
  cohorts: JevCohort[];
  rows: JevOutcomeRow[];
  policy: string;
}

async function jevRequest<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/jev/${path}`, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const value: unknown = await response.json().catch(() => null);
    const detail = value && typeof value === "object" && "detail" in value ? value.detail : null;
    throw new Error(typeof detail === "string" ? detail : `Jev request failed (HTTP ${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const assessJev = (request: JevAssessmentRequest) => jevRequest<JevForecast>("assess", request);
export const fetchJevEvaluation = (signal?: AbortSignal) => requestJson<JevEvaluation>("/api/jev/evaluation", { signal });
export const refreshJevEvaluation = () => jevRequest<JevEvaluation>("evaluation/refresh", {});
