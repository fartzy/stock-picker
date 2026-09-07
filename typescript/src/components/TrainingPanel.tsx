import {
  fetchModelInfo,
  fetchTrainingStatus,
  runTraining,
  type ModelInfoResponse,
  type ThresholdSweepRow,
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

// What's actually loaded for live inference right now -- distinct from
// ModelPicker's "what's chosen for the *next* run" and RunHistory's "what
// past runs looked like." Styled as a labeled meta row (not a trailing
// caption) so it reads as its own fact, not an afterthought glued to the
// run controls above it.
//
// Collapsed by default to a plain "X% ModelA + Y% ModelB" summary -- a
// percentage of the blend reads more intuitively than a raw weight number,
// especially once there's more than one member. The full breakdown (feature
// count per member) is one click away, not shown by default, so this stays
// exactly as glanceable for someone who never clicks it.
function EnsembleComposition({ modelInfo }: { modelInfo: ModelInfoResponse | null }) {
  const [expanded, setExpanded] = useState(false);
  if (!modelInfo || modelInfo.models.length === 0) return null;

  const totalWeight = modelInfo.models.reduce((sum, m) => sum + m.weight, 0);
  const withPct = modelInfo.models.map((m) => ({
    ...m,
    pct: totalWeight > 0 ? (m.weight / totalWeight) * 100 : 0,
  }));

  return (
    <div className="live-model">
      <button
        type="button"
        className="feature-row-toggle"
        onClick={() => setExpanded((e) => !e)}
        title={expanded ? "Hide blend breakdown" : "Show blend breakdown"}
      >
        {expanded ? "▾" : "▸"}
      </button>{" "}
      <span className="meta-label">Live model</span>
      <span className="muted">
        {withPct
          .map((m) => `${m.pct.toFixed(0)}% ${MODEL_TYPE_LABELS[m.model_type] ?? m.model_type}`)
          .join(" + ")}
      </span>
      {expanded && (
        <table className="trade-table" style={{ marginTop: "var(--space-2)" }}>
          <thead>
            <tr>
              <th>Model</th>
              <th className="trade-num">% of blend</th>
              <th className="trade-num">Features</th>
            </tr>
          </thead>
          <tbody>
            {withPct.map((m) => (
              <tr key={m.model_type}>
                <td>{MODEL_TYPE_LABELS[m.model_type] ?? m.model_type}</td>
                <td className="trade-num">{m.pct.toFixed(1)}%</td>
                <td className="trade-num">{m.feature_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// The highlighted threshold row should be the most selective (highest
// threshold) one whose sample is still large enough to trust -- picking
// literally the last populated row could highlight something like "1 trade,
// 100% hit rate," which is noise, not a result. 30 is a common rule-of-thumb
// floor for a binomial rate to mean anything at a glance; this is a headline
// callout, not a rigorous estimate, so it doesn't need to be more precise
// than that.
const HIGHLIGHT_MIN_TRADES = 30;

function pickHighlightRow(sweep: ThresholdSweepRow[]): ThresholdSweepRow | undefined {
  const gated = sweep.filter((row) => row.threshold > 0);
  const qualifying = gated.filter((row) => row.n_trades >= HIGHLIGHT_MIN_TRADES);
  if (qualifying.length > 0) return qualifying[qualifying.length - 1];
  // Nothing cleared the bar (e.g. a small holdout) -- fall back to whichever
  // confidence-gated row has the most trades, still better than the
  // highest-threshold row which could be a sample of one.
  return gated.reduce<ThresholdSweepRow | undefined>(
    (best, row) => (!best || row.n_trades > best.n_trades ? row : best),
    undefined,
  );
}

function TrainingResultSummary({ result }: { result: TrainingResult }) {
  const holdout = result.holdout_metrics;
  const sweep = result.threshold_sweep;
  const highlight = sweep ? pickHighlightRow(sweep) : undefined;
  return (
    <div className="training-result">
      {holdout && (
        <span>
          Holdout: {(holdout.directional_accuracy * 100).toFixed(1)}% accuracy on{" "}
          {holdout.n_test_rows.toLocaleString()} rows.
        </span>
      )}
      {highlight && (
        <span className="muted">
          {" "}
          At {(highlight.threshold * 100).toFixed(1)}% threshold: {highlight.n_trades} trades
          {highlight.avg_picks_per_day != null && ` (~${highlight.avg_picks_per_day.toFixed(1)}/day)`},{" "}
          {(highlight.hit_rate * 100).toFixed(1)}% hit rate.
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
    <div className="training-run-panel">
      <div className="training-controls">
        <button className="btn-primary" onClick={handleRun} disabled={isRunning}>
          {isRunning ? "Training..." : "Run training"}
        </button>
        {/* Idle carries no status text of its own -- Run History below is
            the record of what's happened, this session or any other. */}
        <span className="muted">
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
