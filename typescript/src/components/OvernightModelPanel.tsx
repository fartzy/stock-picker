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
      <p><strong>{data.available ? "Saved next-open model" : "No saved next-open model"}</strong> · Separate from the Rank/Fit open-to-close models.</p>
      {data.available ? (
        <>
          <p>Trained through {data.trained_through}; latest next open observed {data.label_observed_on}. The model uses {data.feature_columns.length} inputs.</p>
          <div className="overnight-model-metrics">
            <span>Held-out model gap MAE <strong>{pct(data.model_gap_mae)}</strong></span>
            <span>Unchanged-price MAE <strong>{pct(data.unchanged_gap_mae)}</strong></span>
            <span>Ticker-mean MAE <strong>{pct(data.ticker_mean_gap_mae)}</strong></span>
            <span>Evaluated rows <strong>{data.evaluated_rows}</strong></span>
          </div>
          {data.model_gap_mae !== null && data.unchanged_gap_mae !== null && data.model_gap_mae >= data.unchanged_gap_mae && (
            <p className="overnight-evidence">Forecasts are available. On held-out sessions, this model had more error than assuming an unchanged open.</p>
          )}
          {!data.serving_inputs_pinned && <p className="overnight-evidence">The saved artifact lacks pinned morning estimators. Retrain it before requesting a forecast.</p>}
        </>
      ) : <p className="muted">The model code and its {data.feature_columns.length} inputs are available for inspection; a forecast requires a saved model. It is never trained during a request.</p>}
      <button className="btn-primary" type="button" onClick={onOpenOvernight}>Open overnight scenario</button>
    </div>
  );
}
