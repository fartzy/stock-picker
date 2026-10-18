import type { ModelFeatures, ModelKey } from "../api";

export type ModelFilter = "all" | ModelKey;

const FILTERS: { id: ModelFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "fit", label: "Fit" },
  { id: "rank", label: "Rank" },
  { id: "overnight", label: "Overnight" },
  { id: "news", label: "News" },
];

// News reads article text; its feature list is what news becomes for Fit/Rank,
// so tagging those columns "News" would double-count them.
const TAGGED_MODELS: ModelKey[] = ["fit", "rank", "overnight"];

/** feature name -> labels of the models that read it. */
export function modelsByFeature(models: ModelFeatures[] | null): Map<string, string[]> {
  const used = new Map<string, string[]>();
  for (const model of models ?? []) {
    if (!TAGGED_MODELS.includes(model.key)) continue;
    for (const feature of model.features) used.set(feature, [...(used.get(feature) ?? []), model.label]);
  }
  return used;
}

export function ModelTags({ feature, usedBy }: { feature: string; usedBy: Map<string, string[]> }) {
  const labels = usedBy.get(feature);
  if (!labels?.length) return null;
  return <>{labels.map((label) => <span key={label} className="feature-badge model-tag">{label}</span>)}</>;
}

export function ModelFilterBar({
  value,
  onChange,
  models,
}: {
  value: ModelFilter;
  onChange: (value: ModelFilter) => void;
  models: ModelFeatures[] | null;
}) {
  const count = (id: ModelFilter) => (id === "all" ? null : models?.find((model) => model.key === id)?.features.length);
  return (
    <div className="meta-row">
      <span className="meta-label">Model</span>
      <div className="interval-toggle">
        {FILTERS.map((filter) => {
          const n = count(filter.id);
          return (
            <button key={filter.id} type="button" className={value === filter.id ? "active" : ""}
              onClick={() => onChange(filter.id)}>
              {filter.label}{n ? ` ${n}` : ""}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** What the news deciders read: article text, not feature columns. */
export function NewsInputs({ model }: { model: ModelFeatures | undefined }) {
  if (!model?.articles) return null;
  return <p className="muted news-inputs">Reads headline + summary, up to {model.articles.max_articles} articles</p>;
}
