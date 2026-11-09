// The app's operational confidence threshold -- matches buy_signal.py's own
// DEFAULT_THRESHOLD. Shared here so BuySignal's input default and
// RunHistory's "this is the threshold we actually bet on" row highlight
// can never drift apart into two different numbers.
export const DEFAULT_BUY_THRESHOLD = 0.005;
// Shared so TrainingPanel's ensemble-composition breakdown and RunHistory's
// past-run cards can never drift out of sync the way they did before --
// each had its own copy of this map, and both were missing neural_net/ridge.
export const MODEL_TYPE_LABELS = {
    lightgbm: "LightGBM",
    lightgbm_rank: "LightGBM Rank",
    random_forest: "Random Forest",
    logistic_regression: "Logistic Regression",
    neural_net: "Neural Net",
    ridge: "Ridge Regression",
};
export const fetchNewsOutcomes = () => getJson("/api/paper-book/news-outcomes");
export const fetchFirstTouch = (cases) => mutate("POST", "/api/paper-book/first-touch", { cases });
export const fetchHoldPath = (asOf, tickers, days) => {
    const params = new URLSearchParams({ as_of: asOf, tickers: tickers.join(","), days: String(days) });
    return getJson(`/api/paper-book/hold-path?${params.toString()}`);
};
async function getJson(path) {
    const response = await fetch(path);
    if (!response.ok) {
        throw new Error(`${path} failed: ${response.status}`);
    }
    return response.json();
}
async function mutate(method, path, body) {
    const response = await fetch(path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
        throw new Error(`${method} ${path} failed: ${response.status}`);
    }
    return response.status === 204 ? undefined : response.json();
}
export const fetchCatalog = () => getJson("/api/catalog");
export const fetchFeatureImportance = () => getJson("/api/feature-importance");
export const fetchModelInfo = () => getJson("/api/model-info");
export const fetchModelTypes = () => getJson("/api/model-types");
export const fetchCoverage = () => getJson("/api/coverage");
export const fetchPriceHistory = (ticker, interval) => getJson(`/api/prices/${encodeURIComponent(ticker)}?interval=${interval}`);
export const fetchFeatureValues = (ticker) => getJson(`/api/features/${encodeURIComponent(ticker)}`);
export const fetchOvernightModel = () => getJson("/api/overnight/model");
export const fetchNewsModels = () => getJson("/api/news/models");
export async function fetchOvernightCurrentPrice(ticker) {
    const response = await fetch(`/api/overnight/current-price?ticker=${encodeURIComponent(ticker)}`);
    if (!response.ok) {
        const body = await response.json().catch(() => null);
        const detail = body && typeof body === "object" && "detail" in body ? String(body.detail) : `HTTP ${response.status}`;
        throw new Error(detail);
    }
    return response.json();
}
export async function forecastOvernight(request) {
    const response = await fetch("/api/overnight/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        const body = await response.json().catch(() => null);
        const detail = body && typeof body === "object" && "detail" in body ? String(body.detail) : `HTTP ${response.status}`;
        throw new Error(detail);
    }
    return response.json();
}
export const fetchModelFeatures = () => getJson("/api/model-features");
export const fetchCorrelation = () => getJson("/api/correlation");
export const fetchRegistry = () => getJson("/api/registry");
export const fetchTrades = () => getJson("/api/trades");
export const fetchPositions = () => getJson("/api/positions");
export const fetchFees = () => getJson("/api/fees");
export const fetchPaperBook = (kind = "both", topK, fitTopK, rankTopK) => {
    const params = new URLSearchParams({ kind });
    if (topK !== undefined)
        params.set("top_k", String(topK));
    if (fitTopK !== undefined)
        params.set("fit_top_k", String(fitTopK));
    if (rankTopK !== undefined)
        params.set("rank_top_k", String(rankTopK));
    return getJson(`/api/paper-book?${params.toString()}`);
};
export const fetchBlacklistedTickers = () => getJson("/api/ticker-blacklist");
export const fetchOvernightActuals = (asOf, tickers) => mutate("POST", "/api/what-if/overnight-actuals", { as_of: asOf, tickers });
export const rebuildPaperBook = () => mutate("POST", "/api/paper-book/rebuild");
export const replayPaperBook = (asOf, modelRunId) => mutate("POST", "/api/paper-book/replay", {
    as_of: asOf,
    model_run_id: modelRunId ?? null,
});
export const fetchBenchmarkReturns = (dates) => getJson(`/api/benchmark-returns?dates=${dates.map(encodeURIComponent).join(",")}`);
export const fetchPrunedFeatures = () => getJson("/api/pruned-features");
export const pruneFeature = (feature, reason) => mutate("POST", `/api/features/${encodeURIComponent(feature)}/prune`, reason ? { reason } : undefined);
export const unpruneFeature = (feature) => mutate("DELETE", `/api/features/${encodeURIComponent(feature)}/prune`);
export const fetchFeatureSelection = () => getJson("/api/feature-selection");
export const setFeatureSelection = (includedFeatures) => mutate("POST", "/api/feature-selection", {
    included_features: includedFeatures,
});
export const clearFeatureSelection = () => mutate("DELETE", "/api/feature-selection");
export const fetchModelSelection = () => getJson("/api/model-selection");
export const setModelSelection = (modelChoices) => mutate("POST", "/api/model-selection", { model_choices: modelChoices });
export const clearModelSelection = () => mutate("DELETE", "/api/model-selection");
export const fetchTrainingStatus = () => getJson("/api/training/status");
export const runTraining = () => mutate("POST", "/api/training/run");
export const fetchMorningCheck = () => getJson("/api/morning-check");
export const fetchMorningCheckRuns = () => getJson("/api/morning-check/runs");
export const runMorningCheck = (which = "both") => mutate("POST", "/api/morning-check", { which });
export const loadMorningCheck = (runId) => mutate("POST", "/api/morning-check/load", { run_id: runId });
export const fetchMorningJob = () => getJson("/api/morning-job");
export const setMorningJob = (enabled) => mutate("PUT", "/api/morning-job", { enabled });
export const fetchMorningScan = () => getJson("/api/morning-scan");
export const runMorningScan = () => mutate("POST", "/api/morning-scan");
export const fetchTrainingRuns = () => getJson("/api/training/runs");
export const fetchLiveModel = () => getJson("/api/live-model");
export const fetchPipelineFreshness = () => getJson("/api/pipeline-freshness");
export const setLiveModel = (runId) => mutate("POST", "/api/live-model", { run_id: runId });
export const resetLiveModel = () => mutate("DELETE", "/api/live-model");
const QUOTE_BATCH = 200;
export async function fetchQuotes(tickers, asOf) {
    const unique = [...new Set(tickers.filter(Boolean))];
    if (unique.length === 0)
        return { quotes: [] };
    const batches = [];
    for (let i = 0; i < unique.length; i += QUOTE_BATCH) {
        batches.push(unique.slice(i, i + QUOTE_BATCH));
    }
    const suffix = asOf ? `&as_of=${encodeURIComponent(asOf)}` : "";
    const parts = await Promise.all(batches.map((batch) => getJson(`/api/quotes?tickers=${batch.map(encodeURIComponent).join(",")}${suffix}`)));
    return { quotes: parts.flatMap((part) => part.quotes) };
}
export const createTrade = (trade) => mutate("POST", "/api/trades", trade);
export const fetchBuySignal = (threshold, live = false, kind = "fit") => getJson(`/api/buy-signal?threshold=${encodeURIComponent(threshold.toString())}${live ? "&live=true" : ""}&kind=${kind}`);
export const fetchUniverse = () => getJson("/api/universe");
