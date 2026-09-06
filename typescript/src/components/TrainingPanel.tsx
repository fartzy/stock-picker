import {
  fetchModelInfo,
  fetchTrainingStatus,
  runTraining,
  type ModelInfoResponse,
  type TrainingResult,
  type TrainingStatusResponse,
} from "../api";
import { useState } from "react";
import { useFetchData } from "../useFetchData";

// Training runs take minutes, not seconds -- poll rather than push, same
// "live-ish via polling" convention the rest of the Feature Store tab uses.
const POLL_INTERVAL_MS = 4000;

const MODEL_TYPE_LABELS: Record<string, string> = {
  lightgbm: "LightGBM",
  random_forest: "Random Forest",
  logistic_regression: "Logistic Regression",
};

function formatTimestamp(isoString: string | null): string {
  if (!isoString) return "";
  return new Date(isoString).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function EnsembleComposition({ modelInfo }: { modelInfo: ModelInfoResponse | null }) {
  if (!modelInfo || modelInfo.models.length === 0) return null;
  return (
    <div className="muted" style={{ fontSize: "var(--text-caption)" }}>
      Ensemble:{" "}
      {modelInfo.models
        .map((m) => {
          const label = MODEL_TYPE_LABELS[m.model_type] ?? m.model_type;
          const role = m.weight === 0 ? "diagnostic only" : `weight ${m.weight}`;
          return `${label} (${role}, ${m.feature_count} features)`;
        })
        .join(" + ")}
    </div>
  );
}

function TrainingResultSummary({ result }: { result: TrainingResult }) {
  const holdout = result.holdout_metrics;
  const sweep = result.threshold_sweep;
  return (
    <div className="training-result">
      {holdout && (
        <span>
          Holdout: {(holdout.directional_accuracy * 100).toFixed(1)}% accuracy on{" "}
          {holdout.n_test_rows.toLocaleString()} rows.
        </span>
      )}
      {sweep && sweep.length > 0 && (
        <span className="muted">
          {sweep
            .filter((row) => row.n_trades > 0)
            .slice(-1)
            .map((row) => (
              <span key={row.threshold}>
                {" "}
                At {(row.threshold * 100).toFixed(1)}% threshold: {row.n_trades} trades,{" "}
                {(row.hit_rate * 100).toFixed(1)}% hit rate.
              </span>
            ))}
        </span>
      )}
    </div>
  );
}

export default function TrainingPanel() {
  const { data, error } = useFetchData<TrainingStatusResponse>(fetchTrainingStatus, {
    intervalMs: POLL_INTERVAL_MS,
  });
  // Composition only actually changes once a run completes, but polling this
  // alongside status is cheap (a single small pickle read) and keeps it in
  // sync without a separate "did status just change" effect.
  const { data: modelInfo } = useFetchData<ModelInfoResponse>(fetchModelInfo, {
    intervalMs: POLL_INTERVAL_MS,
  });
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  async function handleRun() {
    setStarting(true);
    setStartError(null);
    try {
      await runTraining();
    } catch (err) {
      // A 409 just means a run is already in progress elsewhere -- the next
      // poll will show it as "running" regardless, so it isn't a real error.
      const message = err instanceof Error ? err.message : "failed to start training";
      if (!message.includes("409")) setStartError(message);
    } finally {
      setStarting(false);
    }
  }

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading training status...</p>;

  const isRunning = data.status === "running" || starting;

  return (
    <div>
      <div className="training-controls">
        <button className="btn-primary" onClick={handleRun} disabled={isRunning}>
          {isRunning ? "Training..." : "Run training"}
        </button>
        <span className="muted">
          {data.status === "idle" && "No run yet this session."}
          {data.status === "running" && `Running since ${formatTimestamp(data.started_at)}`}
          {data.status === "completed" && `Last completed ${formatTimestamp(data.completed_at)}`}
          {data.status === "failed" && `Last run failed: ${data.error}`}
        </span>
      </div>
      {startError && <p className="error">{startError}</p>}
      {data.status === "completed" && data.result && <TrainingResultSummary result={data.result} />}
      <EnsembleComposition modelInfo={modelInfo} />
    </div>
  );
}
