import { fetchOvernightModel, type OvernightModelResponse } from "../api";
import { OVERNIGHT_FEATURE_DESCRIPTIONS, OVERNIGHT_FEATURE_GROUPS } from "../overnightFeatures";
import { useFetchData } from "../useFetchData";

export default function OvernightFeaturePanel() {
  const { data, error } = useFetchData<OvernightModelResponse>(fetchOvernightModel);
  if (error) return <p className="error">Overnight features: {error}</p>;
  if (!data) return <p className="muted">Loading overnight feature contract…</p>;
  const groups = [...new Set(data.feature_columns.map((name) => OVERNIGHT_FEATURE_GROUPS[name] ?? "Other"))];
  return (
    <details className="view-card overnight-registry">
      <summary><strong>Next-open model features</strong> · {data.feature_columns.length} separate inputs · {data.feature_version}</summary>
      <p className="muted">These do not join Rank/Fit’s feature-selection count. A close assumption changes the close-derived inputs; morning-model outputs are pinned separately.</p>
      {groups.map((group) => (
        <div key={group}>
          <h4>{group}</h4>
          <div className="overnight-feature-grid">
            {data.feature_columns.filter((name) => (OVERNIGHT_FEATURE_GROUPS[name] ?? "Other") === group).map((name) => (
              <div key={name} className="overnight-feature-row">
                <code>{name}</code><small>{OVERNIGHT_FEATURE_DESCRIPTIONS[name] ?? "Model input"}</small>
              </div>
            ))}
          </div>
        </div>
      ))}
    </details>
  );
}
