import {
  fetchCatalog,
  fetchCoverage,
  fetchFeatureImportance,
  fetchRegistry,
  type CatalogResponse,
  type CoverageResponse,
  type FeatureView,
  type ImportanceResponse,
  type RegistryResponse,
} from "../api";
import { coverageColor, importanceColor } from "../theme";
import { useFetchData } from "../useFetchData";

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
  const error =
    [registryError, catalogError, coverageError, importanceError].filter(Boolean).join("; ") || null;

  if (error) return <p className="error">{error}</p>;
  if (!registry || !catalog || !coverage || !importance) return <p className="muted">Loading registry...</p>;

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
      {registry.feature_views.map((view) => (
        <details className="view-card" key={view.name}>
          <summary>
            <strong style={{ color: "var(--accent)" }}>{view.name}</strong>
            <MetaGrid view={view} />
          </summary>
          <div className="view-features">
            {view.features.map((feature) => {
              const pct = coverage.coverage[feature];
              const imp = importance.importance[feature];
              return (
                <div className="feature-row" key={feature}>
                  <div className="feature-row-header">
                    <span>
                      <span style={{ color: coverageColor(pct) }}>&#9679;</span>{" "}
                      <span className="feature-name">{feature}</span>
                    </span>
                    <span className="feature-stats">
                      <span title="Non-null coverage across the universe">
                        {pct !== undefined ? `${Math.round(pct * 100)}% coverage` : "design-only"}
                      </span>
                      <span
                        style={{ color: importanceColor(imp) }}
                        title="Share of the trained model's total LightGBM gain"
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
