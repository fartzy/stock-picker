import { useEffect, useRef, useState } from "react";
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
  type ExperimentalFeature,
  type ExperimentalFeatureView,
  type FeatureSelectionResponse,
  type ImportanceResponse,
  type ModelDerivedFeature,
  type ModelDerivedFeatureView,
  type PrunedFeaturesResponse,
  type RegistryResponse,
} from "../api";
import {
  coverageColor,
  importanceColor,
  IMPORTANCE_GRADIENT_MAX_PCT,
  NEGLIGIBLE_IMPORTANCE_PCT_THRESHOLD,
} from "../theme";
import { experimentalRegistrySections } from "../experimentalRegistry";
import {
  hasUnprunedMarketFeature,
  modelDerivedRegistrySections,
  sameFeatureSet,
  selectableFeatureNames,
} from "../modelDerivedRegistry";
import { useFetchData } from "../useFetchData";
import { FeatureCategory, FeatureRow } from "./FeaturePresentation";

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
  { id: "importance", label: "Highest importance" },
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

function highestImportance(importance: ImportanceResponse): Record<string, number> {
  const result = { ...importance.highest_known };
  for (const [name, value] of Object.entries(importance.importance)) {
    result[name] = Math.max(result[name] ?? 0, value);
  }
  return result;
}

function importanceTitle(name: string, importance: ImportanceResponse): string {
  const current = importance.importance[name];
  return `Highest recorded across saved Fit/Rank models. Current Fit: ${current === undefined ? "excluded" : `${current.toFixed(1)}%`}.`;
}

function ExperimentalViewSection({
  view,
  features,
}: {
  view: ExperimentalFeatureView;
  features: Record<string, ExperimentalFeature>;
}) {
  return (
    <details className="view-card registry-category">
      <summary>
        <strong style={{ color: "var(--accent)" }}>SVM-derived · experimental</strong>
        <span className="muted">{view.features.length} research candidates</span>
      </summary>
      <div className="view-features">
        <p className="muted">{view.description} Values are computed during walk-forward trials.</p>
        {view.features.map((name) => {
          const feature = features[name];
          if (!feature) return null;
          return (
            <details className="feature-row row-hover" key={name}>
              <summary className="feature-row-header">
                <span className="feature-name">{name}</span>
                <span className="muted">model-derived</span>
              </summary>
              <div className="feature-desc">{feature.description}</div>
              <div className="feature-example"><span className="feature-example-label">How computed</span> {feature.computation}</div>
              <div className="feature-example"><span className="feature-example-label">e.g.</span> {feature.example}</div>
            </details>
          );
        })}
      </div>
    </details>
  );
}

function ModelDerivedViewSection({
  view,
  features,
  included,
  pruned,
  importance,
  selectionPending,
  toggleSelected,
  togglePrune,
}: {
  view: ModelDerivedFeatureView;
  features: Record<string, ModelDerivedFeature>;
  included: Set<string> | null;
  pruned: Set<string>;
  importance: ImportanceResponse;
  selectionPending: boolean;
  toggleSelected: (feature: string) => Promise<void>;
  togglePrune: (feature: string, reason?: string) => Promise<void>;
}) {
  return (
    <details className="view-card registry-category">
      <summary>
        <strong style={{ color: "var(--accent)" }}>Model-derived · trainable</strong>
        <span className="muted">{view.features.length} feature{view.features.length === 1 ? "" : "s"}</span>
        {view.features.some((name) => pruned.has(name)) && (
          <span className="category-tally category-tally-pruned">
            {view.features.filter((name) => pruned.has(name)).length} pruned
          </span>
        )}
      </summary>
      <div className="view-features">
        <p className="muted">Fit inputs created during training and scoring; not stored in market-data snapshots.</p>
        {view.features.map((name) => {
          const feature = features[name];
          if (!feature) return null;
          const isPruned = pruned.has(name);
          const imp = highestImportance(importance)[name];
          return (
            <div className="feature-row row-hover" key={name}>
              <div className="feature-row-header">
                <span>
                  <input
                    type="checkbox"
                    checked={included === null || included.has(name)}
                    disabled={selectionPending}
                    onChange={() => void toggleSelected(name)}
                    title="Include this feature in the next training run"
                    aria-label={`Include ${name} in the next training run`}
                  />{" "}
                  <span className={isPruned ? "pruned-feature feature-name" : "feature-name"}>{name}</span>{" "}
                  {isPruned ? (
                    <button type="button" className="feature-badge feature-badge-pruned" onClick={() => void togglePrune(name)}>
                      pruned
                    </button>
                  ) : (
                    <button type="button" className="prune-toggle" onClick={() => void togglePrune(name, "manually pruned from Registry")}>
                      prune
                    </button>
                  )}
                </span>
                <span className="feature-stats">
                  <span title="Created by the companion direction model, not measured from a stored market-data column">model-derived</span>
                  <span style={{ color: importanceColor(imp) }} title={importanceTitle(name, importance)}>
                    {imp !== undefined ? `${imp.toFixed(1)}% peak` : isPruned ? "excluded" : "not trained"}
                  </span>
                </span>
              </div>
              <div className="feature-desc">{feature.description}</div>
              <details className="feature-example">
                <summary>How it works</summary>
                <div>{feature.computation}</div>
                <div>{feature.example}</div>
              </details>
            </div>
          );
        })}
      </div>
    </details>
  );
}

export default function Registry({
  pendingFeature,
  onFeatureFocused,
}: {
  pendingFeature?: string | null;
  onFeatureFocused?: () => void;
} = {}) {
  const { data: registry, error: registryError } = useFetchData<RegistryResponse>(fetchRegistry);
  const { data: catalog, error: catalogError } = useFetchData<CatalogResponse>(fetchCatalog);
  const { data: coverage } = useFetchData<CoverageResponse>(fetchCoverage);
  const { data: importance, error: importanceError } =
    useFetchData<ImportanceResponse>(fetchFeatureImportance);
  const { data: prunedData, error: prunedError } = useFetchData<PrunedFeaturesResponse>(
    fetchPrunedFeatures,
    { intervalMs: PRUNE_POLL_INTERVAL_MS },
  );
  const { data: selectionData, error: selectionError } =
    useFetchData<FeatureSelectionResponse>(fetchFeatureSelection);
  const { data: correlationData } = useFetchData<CorrelationResponse>(fetchCorrelation);
  const [sortMode, setSortMode] = useState<SortMode>("pipeline");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [prunedOverride, setPrunedOverride] = useState<Set<string> | null>(null);
  // undefined = not yet initialized from the fetch; null = no selection
  // (every feature included); Set = an explicit selection.
  const [included, setIncluded] = useState<Set<string> | null | undefined>(undefined);
  const [selectionPending, setSelectionPending] = useState(false);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const [highlightedFeature, setHighlightedFeature] = useState<string | null>(null);
  // Which rows show their description/formula -- collapsed by default so
  // scanning ~97 rows doesn't mean scrolling past that much text for each one.
  const [expandedFeatures, setExpandedFeatures] = useState<Set<string>>(new Set());
  const detailsRefs = useRef(new Map<string, HTMLDetailsElement>());
  const rowRefs = useRef(new Map<string, HTMLDivElement>());

  // One-shot handoff from the Data tab (see App.tsx's pendingFeature): open
  // that feature's category, scroll to its row, and briefly highlight it.
  // Deliberately doesn't call onFeatureFocused() on the "registry not loaded
  // yet" branch below -- a click made before Registry's own fetch resolves
  // should still work once it does, not silently get dropped.
  useEffect(() => {
    // Matches the same loading gate the render below bails out on ("Loading
    // registry...") -- the feature-row/details refs this effect needs don't
    // exist in the DOM until every one of these has resolved, not just
    // `registry` alone (registry often resolves first, before the ref maps
    // below are populated).
    if (
      !pendingFeature ||
      !registry ||
      !catalog ||
      !importance ||
      !prunedData ||
      included === undefined
    ) {
      return;
    }
    const view = registry.feature_views.find((v) => v.features.includes(pendingFeature));
    if (!view) {
      onFeatureFocused?.();
      return;
    }
    const details = detailsRefs.current.get(view.name);
    if (details) details.open = true;
    requestAnimationFrame(() => {
      rowRefs.current.get(pendingFeature)?.scrollIntoView({ behavior: "smooth", block: "center" });
      setHighlightedFeature(pendingFeature);
      // Whatever sent you here (e.g. a Data-tab column link) almost
      // certainly wants the description/formula, not just the row location.
      setExpandedFeatures((prev) => new Set(prev).add(pendingFeature));
      onFeatureFocused?.();
      setTimeout(() => setHighlightedFeature((f) => (f === pendingFeature ? null : f)), 2000);
    });
  }, [
    pendingFeature,
    registry,
    catalog,
    importance,
    prunedData,
    included,
    onFeatureFocused,
  ]);

  const error =
    [registryError, catalogError, importanceError, prunedError, selectionError].filter(Boolean).join("; ") ||
    null;

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
  const allFeatureNames = registry && catalog ? selectableFeatureNames(registry, catalog) : new Set<string>();
  const marketFeatureNames = new Set((registry?.feature_views ?? []).flatMap((view) => view.features));
  const newFeatures = new Set(registry?.new_feature_names ?? []);

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
  if (!registry || !catalog || !importance || !prunedData || included === undefined)
    return <p className="muted">Loading registry...</p>;

  const coverageByFeature = coverage?.coverage ?? {};
  const importanceByFeature = highestImportance(importance);

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
      try {
        await unpruneFeature(feature);
      } catch (error) {
        setMutationError(`Could not unprune ${feature}: ${String(error)}`);
        return;
      }
    } else {
      // Every prune badge (negligible/correlated/plain) triggers pruning
      // directly on click -- someone clicking to see *why* a badge is there
      // (e.g. "correlated to what?") shouldn't silently prune the feature
      // as a side effect of looking.
      if (!window.confirm(`Prune "${feature}" from training?\n\nReason: ${reason ?? "manually pruned"}`)) {
        return;
      }
      next.add(feature);
      try {
        await pruneFeature(feature, reason);
      } catch (error) {
        setMutationError(`Could not prune ${feature}: ${String(error)}`);
        return;
      }
    }
    setMutationError(null);
    setPrunedOverride(next);
  }

  async function toggleSelected(feature: string) {
    if (selectionPending) return;
    // `included === null` means "everything," so the first toggle away from
    // the default has to materialize the full set before removing one name
    // from it -- otherwise there'd be nothing to toggle against.
    const next = new Set(included ?? allFeatureNames);
    if (next.has(feature)) {
      next.delete(feature);
    } else {
      next.add(feature);
    }
    if (!hasUnprunedMarketFeature(next, marketFeatureNames, pruned)) {
      setMutationError("Keep at least one unpruned market-data feature selected; model-derived SVM outputs require raw inputs.");
      return;
    }
    setSelectionPending(true);
    setMutationError(null);
    try {
      const saved = sameFeatureSet(next, allFeatureNames)
        ? await clearFeatureSelection()
        : await setFeatureSelection([...next]);
      setIncluded(saved.included_features ? new Set(saved.included_features) : null);
    } catch (error) {
      setMutationError(`Could not update training selection: ${String(error)}`);
    } finally {
      setSelectionPending(false);
    }
  }

  async function resetSelection() {
    if (selectionPending) return;
    setSelectionPending(true);
    setMutationError(null);
    try {
      const saved = await clearFeatureSelection();
      setIncluded(saved.included_features ? new Set(saved.included_features) : null);
    } catch (error) {
      setMutationError(`Could not reset training selection: ${String(error)}`);
    } finally {
      setSelectionPending(false);
    }
  }

  function toggleExpanded(feature: string) {
    setExpandedFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(feature)) next.delete(feature);
      else next.add(feature);
      return next;
    });
  }

  const selectedCount = included?.size ?? allFeatureNames.size;
  const eligibleCount = [...(included ?? allFeatureNames)].filter(
    (feature) => !pruned.has(feature),
  ).length;

  return (
    <div>
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
        <span className="meta-label">Checked in feature selection</span>
        <span className="muted" style={{ fontSize: "var(--text-caption)" }}>
          {selectedCount} / {allFeatureNames.size}
        </span>
        <span className="muted" style={{ fontSize: "var(--text-caption)" }}>
          {eligibleCount} eligible for automatic training
        </span>
        {included !== null && (
          <button type="button" className="prune-toggle" disabled={selectionPending} onClick={() => void resetSelection()}>
            reset to all
          </button>
        )}
      </div>
      {mutationError && <p className="error" role="alert">{mutationError}</p>}
      {registry.feature_views.map((view) => {
        // A signal before expanding anything -- with ~10 categories and ~97
        // rows total, "what needs attention" shouldn't require opening every
        // one to find out.
        const viewPrunedCount = view.features.filter((f) => pruned.has(f)).length;
        const viewNewCount = view.features.filter((f) => newFeatures.has(f)).length;
        const viewNegligibleCount = view.features.filter((f) => {
          const imp = importanceByFeature[f];
          return !pruned.has(f) && imp !== undefined && imp < NEGLIGIBLE_IMPORTANCE_PCT_THRESHOLD;
        }).length;
        return (
        <FeatureCategory
          title={view.name}
          key={view.name}
          categoryRef={(el) => {
            if (el) detailsRefs.current.set(view.name, el);
          }}
          summaryExtra={<>
            {viewNewCount > 0 && (
              <span className="category-tally category-tally-negligible">{viewNewCount} new</span>
            )}
            {viewPrunedCount > 0 && (
              <span className="category-tally category-tally-pruned">{viewPrunedCount} pruned</span>
            )}
            {viewNegligibleCount > 0 && (
              <span className="category-tally category-tally-negligible">
                {viewNegligibleCount} negligible
              </span>
            )}
            <span className="muted">{view.features.length} inputs</span>
          </>}
        >
            {sortFeatures(view.features, sortMode, sortDirection, coverageByFeature, importanceByFeature).map((feature) => {
              const pct = coverageByFeature[feature];
              const imp = importanceByFeature[feature];
              const isPruned = pruned.has(feature);
              const isNew = newFeatures.has(feature);
              const isNegligible =
                !isPruned && imp !== undefined && imp < NEGLIGIBLE_IMPORTANCE_PCT_THRESHOLD;
              const correlation = !isPruned ? bestCorrelationByFeature[feature] : undefined;
              const isSelected = included === null || included.has(feature);
              const isExpanded = expandedFeatures.has(feature);
              return (
                <FeatureRow
                  key={feature}
                  name={feature}
                  expanded={isExpanded}
                  onToggle={() => toggleExpanded(feature)}
                  highlighted={feature === highlightedFeature}
                  pruned={isPruned}
                  rowRef={(el) => {
                    if (el) rowRefs.current.set(feature, el);
                  }}
                  leading={<>
                      <input
                        type="checkbox"
                        checked={isSelected}
                        disabled={selectionPending}
                        onChange={() => void toggleSelected(feature)}
                        title="Include this feature in the next training run"
                        aria-label={`Include ${feature} in the next training run`}
                      />
                      <span style={{ color: coverageColor(pct) }} aria-hidden="true">&#9679;</span>
                  </>}
                  actions={<>
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
                      {isNew && !isPruned && (
                        <span className="feature-badge">new</span>
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
                          {correlation.correlation.toFixed(2)} vs {correlation.partner}
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
                  </>}
                  stats={<>
                      <span title="Non-null coverage across the universe">
                        {pct !== undefined ? `${Math.round(pct * 100)}% coverage` : "design-only"}
                      </span>
                      <span
                        style={{ color: importanceColor(imp) }}
                        title={importanceTitle(feature, importance)}
                      >
                        {imp !== undefined
                          ? `${imp.toFixed(1)}% peak`
                          : isPruned
                            ? "excluded"
                            : "not trained"}
                      </span>
                      {imp !== undefined && (
                        <span className="importance-bar" title={`${imp.toFixed(1)}% highest recorded importance`}>
                          <span
                            className="importance-bar-fill"
                            style={{
                              width: `${Math.min(100, (imp / IMPORTANCE_GRADIENT_MAX_PCT) * 100)}%`,
                              background: importanceColor(imp),
                            }}
                          />
                        </span>
                      )}
                  </>}
                >
                  <div className="feature-desc">{catalog.descriptions[feature]}</div>
                  <div className="feature-example">
                    <span className="feature-example-label">e.g.</span> {catalog.examples[feature]}
                  </div>
                  <code className="feature-formula">{catalog.formulas[feature]}</code>
                </FeatureRow>
              );
            })}
        </FeatureCategory>
        );
      })}
      {modelDerivedRegistrySections(registry, catalog).map(({ view, features }) => (
        <ModelDerivedViewSection
          key={view.name}
          view={view}
          features={features}
          included={included}
          pruned={pruned}
          importance={importance}
          selectionPending={selectionPending}
          toggleSelected={toggleSelected}
          togglePrune={togglePrune}
        />
      ))}
      {experimentalRegistrySections(registry, catalog).map(({ view, features }) => (
        <ExperimentalViewSection
          key={view.name}
          view={view}
          features={features}
        />
      ))}
    </div>
  );
}
