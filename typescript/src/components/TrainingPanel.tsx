import {
  clearModelSelection,
  fetchModelInfo,
  fetchModelSelection,
  fetchTrainingStatus,
  runTraining,
  setModelSelection,
  type ModelInfoResponse,
  type ModelSelectionResponse,
  type TrainingResult,
  type TrainingStatusResponse,
} from "../api";
import { useEffect, useState } from "react";
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
          return `${label} (weight ${m.weight}, ${m.feature_count} features)`;
        })
        .join(" + ")}
    </div>
  );
}

function ModelPicker({
  selection,
  chosen,
  onToggle,
}: {
  selection: ModelSelectionResponse;
  chosen: Set<string>;
  onToggle: (modelType: string) => void;
}) {
  return (
    <div className="meta-row">
      <span className="meta-label">Ensemble models</span>
      {selection.available_model_types.map((modelType) => (
        <label key={modelType} className="chip">
          <input type="checkbox" checked={chosen.has(modelType)} onChange={() => onToggle(modelType)} />{" "}
          {MODEL_TYPE_LABELS[modelType] ?? modelType}
        </label>
      ))}
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
  const { data: modelSelection } = useFetchData<ModelSelectionResponse>(fetchModelSelection);
  // undefined = not yet initialized from the fetch; null = no explicit
  // choice (every available model type included); Set = an explicit choice.
  // Nothing else in the app mutates this concurrently, so (like Registry's
  // feature selection) it's seeded once and then owned locally.
  const [chosenModelTypes, setChosenModelTypes] = useState<Set<string> | null | undefined>(undefined);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  useEffect(() => {
    if (chosenModelTypes === undefined && modelSelection) {
      setChosenModelTypes(
        modelSelection.model_choices ? new Set(modelSelection.model_choices.map((c) => c.model_type)) : null,
      );
    }
  }, [modelSelection, chosenModelTypes]);

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

  async function toggleModelType(modelType: string) {
    if (!modelSelection) return;
    const next = new Set(chosenModelTypes ?? modelSelection.available_model_types);
    if (next.has(modelType)) {
      // Always leave at least one model type selected -- an empty ensemble
      // has nothing to blend and nothing to train.
      if (next.size === 1) return;
      next.delete(modelType);
    } else {
      next.add(modelType);
    }
    if (next.size === modelSelection.available_model_types.length) {
      setChosenModelTypes(null);
      await clearModelSelection();
    } else {
      setChosenModelTypes(next);
      await setModelSelection([...next].map((type) => ({ model_type: type, weight: 1.0 })));
    }
  }

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading training status...</p>;

  const isRunning = data.status === "running" || starting;
  const chosen = modelSelection ? (chosenModelTypes ?? new Set(modelSelection.available_model_types)) : null;

  return (
    <div>
      {modelSelection && chosen && (
        <ModelPicker selection={modelSelection} chosen={chosen} onToggle={toggleModelType} />
      )}
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
