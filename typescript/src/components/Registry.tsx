import { fetchCatalog, fetchCoverage, fetchRegistry, type CatalogResponse, type CoverageResponse, type RegistryResponse } from "../api";
import { coverageColor } from "../theme";
import { useFetchData } from "../useFetchData";

export default function Registry() {
  const { data: registry, error: registryError } = useFetchData<RegistryResponse>(fetchRegistry);
  const { data: catalog, error: catalogError } = useFetchData<CatalogResponse>(fetchCatalog);
  const { data: coverage, error: coverageError } = useFetchData<CoverageResponse>(fetchCoverage);
  const error =
    [registryError, catalogError, coverageError].filter(Boolean).join("; ") || null;

  if (error) return <p className="error">{error}</p>;
  if (!registry || !catalog || !coverage) return <p className="muted">Loading registry...</p>;

  return (
    <div>
      <div className="meta-row">
        <span className="meta-label">Entities</span>
        {registry.entities.map((e) => (
          <span className="chip" key={e.name}>
            {e.name}
          </span>
        ))}
      </div>
      <div className="meta-row">
        <span className="meta-label">Feature services</span>
        {registry.feature_services.map((s) => (
          <span className="chip" key={s.name}>
            {s.name}
          </span>
        ))}
      </div>
      {registry.feature_views.map((view) => (
        <details className="view-card" key={view.name}>
          <summary>
            <strong style={{ color: "var(--accent)" }}>{view.name}</strong>{" "}
            <span className="view-meta">
              source: {view.source} &middot; ttl: {view.ttl_days}d &middot; owner: {view.owner} &middot;{" "}
              {view.features.length} features
              {Object.keys(view.tags).length > 0 &&
                ` · tags: ${Object.entries(view.tags)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(", ")}`}
            </span>
          </summary>
          <div className="view-features">
            {view.features.map((feature) => {
              const pct = coverage.coverage[feature];
              return (
                <div className="feature-row" key={feature}>
                  <span style={{ color: coverageColor(pct) }}>&#9679;</span>{" "}
                  <span className="feature-name">{feature}</span>{" "}
                  <span className="muted">{pct !== undefined ? `${Math.round(pct * 100)}%` : "design-only"}</span>
                  <div className="feature-desc">{catalog.descriptions[feature]}</div>
                </div>
              );
            })}
          </div>
        </details>
      ))}
    </div>
  );
}
