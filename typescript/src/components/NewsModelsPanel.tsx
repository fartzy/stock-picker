import { fetchNewsModels, type NewsModelInfo, type NewsModelsResponse } from "../api";
import { useFetchData } from "../useFetchData";
import NewsFlagOutcome from "./NewsFlagOutcome";

// Judge label the API uses when no model decided a check.
const NO_JUDGE = "none";

const DETAIL_LABELS: Record<string, string> = {
  model: "Model",
  endpoint: "Endpoint",
  key_configured: "API key",
  timeout_seconds: "Timeout",
  labeled_headlines: "Labeled headlines",
  event_phrases: "Event phrases",
};

function detailValue(key: string, value: string | number | boolean | null): string {
  if (key === "key_configured") return value ? "Configured" : "Missing";
  if (key === "timeout_seconds") return `${value}s`;
  return value === null ? "Not set" : String(value);
}

function share(part: number, whole: number): string {
  return whole ? `${Math.round((part / whole) * 100)}%` : "No checks";
}

function NewsModelCard({ model, usage, judged }: { model: NewsModelInfo; usage: NewsModelsResponse["usage"]; judged: number }) {
  const counts = usage.by_judge[model.key] ?? { decided: 0, flagged: 0 };
  const keyMissing = model.key === "grok" && model.detail.key_configured === false;
  return (
    <div className="news-model-card">
      <div className="overnight-model-head">
        <div>
          <strong>{model.name}</strong>
          <p className="overnight-model-meta">{model.kind} · {model.source}</p>
        </div>
        <span className="overnight-badge">{keyMissing ? "Key missing" : "Live"}</span>
      </div>
      <p className="overnight-model-meta">{model.role}</p>
      <div className="overnight-model-metrics">
        <span><small>Decided</small><strong>{counts.decided}</strong></span>
        <span><small>Flagged</small><strong>{counts.flagged}</strong></span>
        <span><small>Share of judged picks</small><strong>{share(counts.decided, judged)}</strong></span>
      </div>
      {Object.keys(model.detail).length > 0 && (
        <dl className="news-model-detail">
          {Object.entries(model.detail).map(([key, value]) => (
            <div key={key}><dt>{DETAIL_LABELS[key] ?? key}</dt><dd>{detailValue(key, value)}</dd></div>
          ))}
        </dl>
      )}
    </div>
  );
}

/** The news deciders that gate picks. None has a measured accuracy, so this
 * shows configuration and how often each one decided in recent scans. */
export default function NewsModelsPanel() {
  const { data, error } = useFetchData<NewsModelsResponse>(fetchNewsModels);
  if (error) return <p className="error">News models: {error}</p>;
  if (!data) return <p className="muted">Loading news models…</p>;
  const { usage, scan_days: days } = data;
  // Checks with no judge never reached a news model (not checked, or no news).
  const unjudged = usage.by_judge[NO_JUDGE]?.decided ?? 0;
  const judged = usage.checked - unjudged;
  const range = days.length ? `${days[0]} to ${days[days.length - 1]}` : "no saved scans";
  return (
    <div className="overnight-model-panel">
      <NewsFlagOutcome />
      <p className="overnight-model-meta">
        Run in order on every pick: event guard, then Grok, then the classifier if Grok is unavailable.
        {" "}{judged} picks judged over {days.length} scan days ({range}) · {usage.flagged} flagged · {unjudged} never reached a model
        {usage.llm_unavailable > 0 && ` · Grok unavailable on ${usage.llm_unavailable}`}
      </p>
      {data.models.map((model) => <NewsModelCard key={model.key} model={model} usage={usage} judged={judged} />)}
      <p className="overnight-model-caution">No accuracy is recorded for these models yet; counts come from saved morning scans.</p>
    </div>
  );
}
