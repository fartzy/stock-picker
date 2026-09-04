import { fetchRegistry, type RegistryResponse } from "../api";
import { useFetchData } from "../useFetchData";

export default function Registry() {
  const { data, error } = useFetchData<RegistryResponse>(fetchRegistry);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading registry...</p>;

  return (
    <div>
      <div className="muted" style={{ marginBottom: 10 }}>
        Entities: {data.entities.map((e) => e.name).join(", ")}
      </div>
      <div className="muted" style={{ marginBottom: 16 }}>
        Feature services: {data.feature_services.map((s) => s.name).join(", ")}
      </div>
      {data.feature_views.map((view) => (
        <div className="view-card" key={view.name}>
          <strong style={{ color: "var(--accent)" }}>{view.name}</strong>
          <div className="view-meta">
            source: {view.source} &middot; ttl: {view.ttl_days}d &middot; owner: {view.owner} &middot;{" "}
            {view.features.length} features
            {Object.keys(view.tags).length > 0 &&
              ` · tags: ${Object.entries(view.tags)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ")}`}
          </div>
        </div>
      ))}
    </div>
  );
}
