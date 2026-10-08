import { useState } from "react";
import { fetchOvernightModel, type OvernightModelResponse } from "../api";
import { OVERNIGHT_FEATURE_DESCRIPTIONS, OVERNIGHT_FEATURE_GROUPS } from "../overnightFeatures";
import { useFetchData } from "../useFetchData";
import { FeatureCategory, FeatureRow } from "./FeaturePresentation";

export default function OvernightFeaturePanel() {
  const { data, error } = useFetchData<OvernightModelResponse>(fetchOvernightModel);
  const [expandedFeatures, setExpandedFeatures] = useState<Set<string>>(new Set());
  if (error) return <p className="error">Next-open model inputs: {error}</p>;
  if (!data) return <p className="muted">Loading next-open model inputs…</p>;
  const groups = [...new Set(data.feature_columns.map((name) => OVERNIGHT_FEATURE_GROUPS[name] ?? "Other"))];
  function toggleExpanded(name: string) {
    setExpandedFeatures((previous) => {
      const next = new Set(previous);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }
  return (
    <details className="view-card overnight-registry">
      <summary><strong>Next-open model features</strong> · {data.feature_columns.length} separate inputs</summary>
      <p className="muted">These do not join Rank/Fit’s feature-selection count. A close assumption changes the close-derived inputs; morning-model outputs are pinned separately.</p>
      {groups.map((group) => {
        const columns = data.feature_columns.filter((name) => (OVERNIGHT_FEATURE_GROUPS[name] ?? "Other") === group);
        return (
          <FeatureCategory
            title={group}
            summaryExtra={<span className="view-meta">{columns.length} inputs</span>}
            className="overnight-feature-category"
            key={group}
          >
            {columns.map((name) => (
              <FeatureRow key={name} name={name} expanded={expandedFeatures.has(name)} onToggle={() => toggleExpanded(name)}>
                <div className="feature-desc">{OVERNIGHT_FEATURE_DESCRIPTIONS[name] ?? "Model input"}</div>
              </FeatureRow>
            ))}
          </FeatureCategory>
        );
      })}
    </details>
  );
}
