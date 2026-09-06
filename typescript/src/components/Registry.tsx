import { useEffect, useState } from "react";
import {
  clearFeatureSelection,
  fetchCatalog,
  fetchCorrelation,
  fetchCoverage,
  fetchFeatureImportance,
  fetchFeatureSelection,
  fetchPrunedFeatures,
  fetchRegistry,
  pruneFeature,
  setFeatureSelection,
  unpruneFeature,
  type CatalogResponse,
  type CorrelationResponse,
  type CoverageResponse,
  type FeatureSelectionResponse,
  type FeatureView,
  type ImportanceResponse,
  type PrunedFeaturesResponse,
  type RegistryResponse,
} from "../api";
import { coverageColor, importanceColor, NEGLIGIBLE_IMPORTANCE_PCT_THRESHOLD } from "../theme";
import { useFetchData } from "../useFetchData";

// Prunes can also happen from CorrelationHeatmap (a sibling tab section) --
// poll rather than fetch-once so a prune made there shows up here too, same
// "live-ish via polling" convention PruneArchive used before folding into
// Registry.
const PRUNE_POLL_INTERVAL_MS = 4000;

type SortMode = "pipeline" | "coverage" | "importance";
type SortDirection = "asc" | "desc";

const SORT_MODES: { id: SortMode; label: string }[] = [
  { id: "pipeline", label: "Pipeline order" },
  { id: "coverage", label: "Coverage" },
  { id: "importance", label: "Importance" },
];

// Coverage/importance are attributes of a feature, not a separate view -- this
// sorts each category's own feature list rather than adding a whole standalone
// "worst first" screen duplicating what's already shown per-feature below.
// Missing values (e.g. design-only cross-sectional columns with no coverage
// entry) always sort to the end, regardless of direction -- they aren't a
// "worst" or "best" case, just not applicable.
function sortFeatures(
  features: string[],
  mode: SortMode,
  direction: SortDirection,
  coverage: Record<string, number>,
  importance: Record<string, number>,
): string[] {
  if (mode === "pipeline") return features;
  const values = mode === "coverage" ? coverage : importance;
  const missingValue = direction === "asc" ? Infinity : -Infinity;
  return [...features].sort((a, b) => {
    const va = values[a] ?? missingValue;
    const vb = values[b] ?? missingValue;
    return direction === "asc" ? va - vb : vb - va;
  });
}

const MODEL_TYPE_LABELS: Record<string, string> = {
  lightgbm: "LightGBM",
  random_forest: "RF",
  logistic_regression: "LogReg",
};

// The blended `importance` number is a weighted average across ensemble
// members -- useful as a single sortable number, but it hides how each model
// type actually sees a feature (e.g. logistic_regression's coefficient-based
// view is a diagnostic-only member with zero weight in the blend, so its
// value never shows up there at all). This tooltip surfaces the breakdown.
function formatImportanceBreakdown(
  feature: string,
  byModelType: Record<string, Record<string, number>> | undefined,
): string {
  if (!byModelType) return "Share of the trained ensemble's blended importance";
  const parts = Object.entries(byModelType)
    .map(([modelType, values]) => {
      const value = values[feature];
      if (value === undefined) return null;
      const label = MODEL_TYPE_LABELS[modelType] ?? modelType;
      return `${label}: ${value.toFixed(1)}%`;
    })
    .filter((part): part is string => part !== null);
  return parts.length > 0 ? parts.join(" · ") : "Share of the trained ensemble's blended importance";
}

// Service/entity names are snake_case identifiers (e.g. "day_session_return_model")
// -- fine as a technical name, messy sitting next to a Title Case section label.
// Display-only: the raw name is still what's used as the React key.
function humanize(identifier: string): string {
  return identifier
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function MetaGrid({ view }: { view: FeatureView }) {
  return (
    <div className="view-metagrid">
      <div className="meta-item">
        <span className="meta-key">Source</span>
        <span className="meta-value">{view.source}</span>
      </div>
      <div className="meta-item">
        <span className="meta-key">TTL</span>
        <span className="meta-value">{view.ttl_days}d</span>
      </div>
      <div className="meta-item">
        <span className="meta-key">Owner</span>
        <span className="meta-value">{view.owner}</span>
      </div>
      <div className="meta-item">
        <span className="meta-key">Features</span>
        <span className="meta-value">{view.features.length}</span>
      </div>
      {Object.entries(view.tags).map(([key, value]) => (
        <div className="meta-item" key={key}>
          <span className="meta-key">{key}</span>
          <span className="meta-value">{value}</span>
        </div>
      ))}
    </div>
  );
}

export default function Registry() {
  const { data: registry, error: registryError } = useFetchData<RegistryResponse>(fetchRegistry);
  const { data: catalog, error: catalogError } = useFetchData<CatalogResponse>(fetchCatalog);
  const { data: coverage, error: coverageError } = useFetchData<CoverageResponse>(fetchCoverage);
  const { data: importance, error: importanceError } =
    useFetchData<ImportanceResponse>(fetchFeatureImportance);
  const { data: prunedData, error: prunedError } = useFetchData<PrunedFeaturesResponse>(
    fetchPrunedFeatures,
    { intervalMs: PRUNE_POLL_INTERVAL_MS },
  );
  const { data: selectionData, error: selectionError } =
    useFetchData<FeatureSelectionResponse>(fetchFeatureSelection);
  const { data: correlationData, error: correlationError } =
    useFetchData<CorrelationResponse>(fetchCorrelation);
  const [sortMode, setSortMode] = useState<SortMode>("pipeline");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [prunedOverride, setPrunedOverride] = useState<Set<string> | null>(null);
  // undefined = not yet initialized from the fetch; null = no selection
  // (every feature included); Set = an explicit selection.
  const [included, setIncluded] = useState<Set<string> | null | undefined>(undefined);
  const error =
    [registryError, catalogError, coverageError, importanceError, prunedError, selectionError, correlationError]
      .filter(Boolean)
      .join("; ") || null;

  const pruned = prunedOverride ?? new Set(prunedData?.pruned_features ?? []);
  const reasonByFeature = Object.fromEntries(
    (prunedData?.archive ?? []).map((entry) => [entry.feature, entry.reason]),
  );
  // Each feature's single strongest partner from the correlation top-pairs
  // list (a global top-30, not per-feature -- most features won't appear in
  // it at all, which is fine, they just get no badge).
  const bestCorrelationByFeature: Record<string, { partner: string; correlation: number }> = {};
  for (const pair of correlationData?.top_pairs ?? []) {
    for (const [feature, partner] of [
      [pair.a, pair.b],
      [pair.b, pair.a],
    ] as const) {
      const existing = bestCorrelationByFeature[feature];
      if (!existing || Math.abs(existing.correlation) < Math.abs(pair.correlation)) {
        bestCorrelationByFeature[feature] = { partner, correlation: pair.correlation };
      }
    }
  }
  const allFeatureNames = new Set(
    (registry?.feature_views ?? []).flatMap((view) => view.features),
  );

  // Once the poll confirms the override matches the server, drop it and
  // trust the poll again -- otherwise a prune made elsewhere (e.g. the
  // Correlation tab) would never show up here once any local toggle happened.
  useEffect(() => {
    if (!prunedOverride || !prunedData) return;
    const serverSet = new Set(prunedData.pruned_features);
    const matches =
      prunedOverride.size === serverSet.size && [...prunedOverride].every((f) => serverSet.has(f));
    if (matches) setPrunedOverride(null);
  }, [prunedData, prunedOverride]);

  // Local selection state is seeded once from the initial fetch, then owned
  // locally -- unlike pruning, nothing else in the app mutates the selection
  // concurrently, so there's no need to poll or reconcile against the server.
  useEffect(() => {
    if (included === undefined && selectionData) {
      setIncluded(selectionData.included_features ? new Set(selectionData.included_features) : null);
    }
  }, [selectionData, included]);

  if (error) return <p className="error">{error}</p>;
  if (
    !registry ||
    !catalog ||
    !coverage ||
    !importance ||
    !prunedData ||
    !correlationData ||
    included === undefined
  )
    return <p className="muted">Loading registry...</p>;

  // Clicking the already-active mode flips direction (like a sortable table
  // header); clicking a different mode switches to it at a sensible default
  // (worst-first for coverage, best-first for importance).
  function handleSortClick(mode: SortMode) {
    if (mode === sortMode) {
      setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortMode(mode);
      setSortDirection(mode === "importance" ? "desc" : "asc");
    }
  }

  async function togglePrune(feature: string, reason?: string) {
    const next = new Set(pruned);
    if (next.has(feature)) {
      next.delete(feature);
      await unpruneFeature(feature);
    } else {
      next.add(feature);
      await pruneFeature(feature, reason);
    }
    setPrunedOverride(next);
  }

  async function toggleSelected(feature: string) {
    // `included === null` means "everything," so the first toggle away from
    // the default has to materialize the full set before removing one name
    // from it -- otherwise there'd be nothing to toggle against.
    const next = new Set(included ?? allFeatureNames);
    if (next.has(feature)) {
      next.delete(feature);
    } else {
      next.add(feature);
    }
    if (next.size === allFeatureNames.size) {
      setIncluded(null);
      await clearFeatureSelection();
    } else {
      setIncluded(next);
      await setFeatureSelection([...next]);
    }
  }

  async function resetSelection() {
    setIncluded(null);
    await clearFeatureSelection();
  }

  const selectedCount = included?.size ?? allFeatureNames.size;

  return (
    <div>
      <div className="meta-row">
        <span className="meta-label">Entities</span>
        {registry.entities.map((e) => (
          <span className="chip" key={e.name}>
            {humanize(e.name)}
          </span>
        ))}
      </div>
      <div className="meta-row">
        <span className="meta-label">Feature services</span>
        {registry.feature_services.map((s) => (
          <span className="chip" key={s.name} title={s.name}>
            {humanize(s.name)}
          </span>
        ))}
      </div>
      <div className="meta-row">
        <span className="meta-label">Sort features by</span>
        <div className="interval-toggle">
          {SORT_MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              className={sortMode === m.id ? "active" : ""}
              onClick={() => handleSortClick(m.id)}
            >
              {m.label}
              {sortMode === m.id && m.id !== "pipeline" && (sortDirection === "asc" ? " ▲" : " ▼")}
            </button>
          ))}
        </div>
        {sortMode !== "pipeline" && (
          <span className="muted" style={{ fontSize: "var(--text-caption)" }}>
            {sortMode === "coverage" ? "Coverage" : "Importance"},{" "}
            {sortDirection === "asc" ? "lowest first" : "highest first"}. Click again to flip.
          </span>
        )}
      </div>
      <div className="meta-row">
        <span className="meta-label">Selected for training</span>
        <span className="muted" style={{ fontSize: "var(--text-caption)" }}>
          {selectedCount} / {allFeatureNames.size}
        </span>
        {included !== null && (
          <button type="button" className="prune-toggle" onClick={resetSelection}>
            reset to all
          </button>
        )}
      </div>
      {registry.feature_views.map((view) => (
        <details className="view-card" key={view.name}>
          <summary>
            <strong style={{ color: "var(--accent)" }}>{view.name}</strong>
            <MetaGrid view={view} />
          </summary>
          <div className="view-features">
            {sortFeatures(view.features, sortMode, sortDirection, coverage.coverage, importance.importance).map((feature) => {
              const pct = coverage.coverage[feature];
              const imp = importance.importance[feature];
              const isPruned = pruned.has(feature);
              const isNegligible =
                !isPruned && imp !== undefined && imp < NEGLIGIBLE_IMPORTANCE_PCT_THRESHOLD;
              const correlation = !isPruned ? bestCorrelationByFeature[feature] : undefined;
              const isSelected = included === null || included.has(feature);
              return (
                <div className="feature-row" key={feature}>
                  <div className="feature-row-header">
                    <span>
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleSelected(feature)}
                        title="Include this feature in the next training run"
                      />{" "}
                      <span style={{ color: coverageColor(pct) }}>&#9679;</span>{" "}
                      <span className={isPruned ? "pruned-feature feature-name" : "feature-name"}>
                        {feature}
                      </span>{" "}
                      {isPruned && (
                        <button
                          type="button"
                          className="feature-badge feature-badge-pruned"
                          onClick={() => togglePrune(feature)}
                          title={reasonByFeature[feature] ?? "manually pruned"}
                        >
                          pruned
                        </button>
                      )}
                      {isNegligible && (
                        <button
                          type="button"
                          className="feature-badge feature-badge-negligible"
                          onClick={() => togglePrune(feature, `negligible importance (${imp.toFixed(2)}%)`)}
                          title="Not moving the model. Click to prune."
                        >
                          negligible
                        </button>
                      )}
                      {correlation && (
                        <button
                          type="button"
                          className="feature-badge feature-badge-correlated"
                          onClick={() =>
                            togglePrune(
                              feature,
                              `high correlation to ${correlation.partner} (r=${correlation.correlation.toFixed(3)})`,
                            )
                          }
                          title={`Correlated with ${correlation.partner} (r=${correlation.correlation.toFixed(3)}). Click to prune.`}
                        >
                          corr {correlation.correlation >= 0 ? "+" : ""}
                          {correlation.correlation.toFixed(2)}
                        </button>
                      )}
                      {!isPruned && !isNegligible && !correlation && (
                        <button
                          type="button"
                          className="prune-toggle"
                          onClick={() => togglePrune(feature, "manually pruned from Registry")}
                        >
                          prune
                        </button>
                      )}
                    </span>
                    <span className="feature-stats">
                      <span title="Non-null coverage across the universe">
                        {pct !== undefined ? `${Math.round(pct * 100)}% coverage` : "design-only"}
                      </span>
                      <span
                        style={{ color: importanceColor(imp) }}
                        title={formatImportanceBreakdown(feature, importance.by_model_type)}
                      >
                        {imp !== undefined ? `${imp.toFixed(1)}% importance` : "--"}
                      </span>
                    </span>
                  </div>
                  <div className="feature-desc">{catalog.descriptions[feature]}</div>
                  <code className="feature-formula">{catalog.formulas[feature]}</code>
                </div>
              );
            })}
          </div>
        </details>
      ))}
    </div>
  );
}
