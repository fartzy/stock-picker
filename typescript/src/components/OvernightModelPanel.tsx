import { fetchOvernightModel, type OvernightModelResponse } from "../api";
import { useFetchData } from "../useFetchData";

function pct(value: number | null): string {
  return value === null ? "Not measured" : `${(value * 100).toFixed(3)}%`;
}

export default function OvernightModelPanel({ onOpenOvernight }: { onOpenOvernight: () => void }) {
  const { data, error } = useFetchData<OvernightModelResponse>(fetchOvernightModel);
  if (error) return <p className="error">Overnight model: {error}</p>;
  if (!data) return <p className="muted">Loading overnight model…</p>;
  return (
    <div className="overnight-model-panel">
      <div className="overnight-model-head">
        <div>
          <strong>Next-open forecast</strong>
          <span>Separate from Rank and Fit</span>
          {data.available && (
            <p className="overnight-model-meta">{data.feature_columns.length} inputs · Trained through {data.trained_through} · {data.evaluated_rows} evaluated rows</p>
          )}
        </div>
        <span className="overnight-badge">{data.available ? "Saved model" : "Not trained"}</span>
      </div>
      {data.available ? (
        <>
          <div className="overnight-model-metrics">
            <span><small>Model error</small><strong>{pct(data.model_gap_mae)}</strong></span>
            <span><small>No-change baseline</small><strong>{pct(data.unchanged_gap_mae)}</strong></span>
            <span><small>Ticker baseline</small><strong>{pct(data.ticker_mean_gap_mae)}</strong></span>
          </div>
        </>
      ) : <p className="overnight-model-meta">{data.feature_columns.length} inputs defined · Save a model to forecast</p>}
      <div className="overnight-model-footer">
        <div>
          {data.available && data.model_gap_mae !== null && data.unchanged_gap_mae !== null && data.model_gap_mae >= data.unchanged_gap_mae && (
            <p className="overnight-model-caution">Historical model error exceeded the no-change baseline.</p>
          )}
          {data.available && !data.serving_inputs_pinned && (
            <p className="overnight-model-caution">Morning inputs are not pinned; retrain before forecasting.</p>
          )}
        </div>
        <button className="btn-primary" type="button" onClick={onOpenOvernight}>Open overnight scenario</button>
      </div>
    </div>
  );
}
