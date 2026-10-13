import { fetchResearchModels, type ResearchModelsResponse } from "../api";
import { useFetchData } from "../useFetchData";

function label(name: string): string {
  if (name.startsWith("rank_gain_prune_5_full_")) return "Rank · 5 low-gain inputs removed";
  if (name.startsWith("rank_gain_prune_3_full_")) return "Rank · 3 low-gain inputs removed";
  if (name.startsWith("rank_all_full_")) return "Rank · all features, full history";
  if (name.startsWith("rank_all_")) return "Rank · all features, recent history";
  if (name.startsWith("rank_breadth_price_")) return "Rank · breadth-price trial";
  return name.replaceAll("_", " ");
}

export default function ResearchModels() {
  const { data, error } = useFetchData<ResearchModelsResponse>(fetchResearchModels);
  if (error) return <p className="error">Research models unavailable: {error}</p>;
  if (!data) return <p className="muted">Loading research models...</p>;
  return (
    <div>
      <p className="muted" style={{ marginTop: 0 }}>
        Saved experiments, separate from Training History. They do not power morning picks.
      </p>
      {data.models.length === 0 && <p className="muted">No research models saved here.</p>}
      {data.models.map((model) => (
        <div className="model-type-row row-hover" key={model.name}>
          <strong>{label(model.name)}</strong>
          <div className="model-type-caption">
            {model.feature_count} inputs · {model.new_feature_count} new price-pattern inputs · research only
          </div>
        </div>
      ))}
    </div>
  );
}
