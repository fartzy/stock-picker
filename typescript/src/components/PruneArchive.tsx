import { useEffect, useState } from "react";
import { fetchPrunedFeatures, unpruneFeature, type PrunedFeaturesResponse } from "../api";
import { useFetchData } from "../useFetchData";

// Polls rather than pushing: prunes can happen from a sibling component
// (CorrelationHeatmap), so this needs to notice changes made elsewhere --
// same "live-ish via polling, not a websocket" convention useFetchData
// already documents.
const POLL_INTERVAL_MS = 4000;

function formatPrunedAt(isoString: string): string {
  return new Date(isoString).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function PruneArchive() {
  const { data, error } = useFetchData<PrunedFeaturesResponse>(fetchPrunedFeatures, {
    intervalMs: POLL_INTERVAL_MS,
  });
  const [restoredLocally, setRestoredLocally] = useState<Set<string>>(new Set());

  // Once the poll confirms a restored feature is really gone, drop it from the
  // local exclusion set -- otherwise re-pruning the same feature later would
  // stay hidden here forever.
  useEffect(() => {
    if (!data || restoredLocally.size === 0) return;
    const stillArchived = new Set(data.archive.map((entry) => entry.feature));
    const stale = [...restoredLocally].filter((feature) => !stillArchived.has(feature));
    if (stale.length > 0) {
      setRestoredLocally((prev) => {
        const next = new Set(prev);
        stale.forEach((feature) => next.delete(feature));
        return next;
      });
    }
  }, [data, restoredLocally]);

  async function restore(feature: string) {
    setRestoredLocally((prev) => new Set(prev).add(feature));
    await unpruneFeature(feature);
  }

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading archive...</p>;

  const archive = data.archive.filter((entry) => !restoredLocally.has(entry.feature));

  if (archive.length === 0) {
    return <p className="muted">No features pruned yet.</p>;
  }

  return (
    <div className="view-card" style={{ padding: "var(--space-3)" }}>
      {archive.map((entry) => (
        <div className="archive-row" key={entry.feature}>
          <div>
            <span className="pruned-feature feature-name">{entry.feature}</span>
            <div className="feature-desc">{entry.reason}</div>
          </div>
          <div className="archive-row-actions">
            <span className="muted" style={{ fontSize: "var(--text-caption)" }}>
              {formatPrunedAt(entry.pruned_at)}
            </span>
            <button className="prune-toggle" onClick={() => restore(entry.feature)}>
              restore
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
