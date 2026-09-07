import {
  DEFAULT_BUY_THRESHOLD,
  fetchTrainingRuns,
  type FoldMetrics,
  type RunModelSpec,
  type ThresholdSweepRow,
  type TrainingRunRecord,
  type TrainingRunsResponse,
} from "../api";
import { useFetchData } from "../useFetchData";

// Row equality tolerance for a value that traveled through JSON as a float --
// exact match would work today, but this is cheap insurance against a
// backend rounding change silently breaking the highlight.
const THRESHOLD_MATCH_EPSILON = 1e-9;

const MODEL_TYPE_LABELS: Record<string, string> = {
  lightgbm: "LightGBM",
  random_forest: "Random Forest",
  logistic_regression: "Logistic Regression",
};

function formatTimestamp(isoString: string): string {
  return new Date(isoString).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60);
  return `${minutes}m ${remaining}s`;
}

function formatComposition(specs: RunModelSpec[]): string {
  return specs.map((s) => `${MODEL_TYPE_LABELS[s.model_type] ?? s.model_type} (weight ${s.weight})`).join(" + ");
}

// Full list on hover via `title` rather than rendered inline -- a run can
// span hundreds of tickers/features, and this is drill-down detail, not
// the headline.
function CountWithTooltip({ items, label }: { items: string[]; label: string }) {
  return (
    <span title={items.join(", ")}>
      {items.length} {label}
    </span>
  );
}

function MetricsTable({ rows }: { rows: (FoldMetrics | ThresholdSweepRow)[] }) {
  if (rows.length === 0) return null;
  const columns = Object.keys(rows[0]);
  return (
    <table className="trade-table">
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => {
          // Rows are flat records of number/string metrics -- reflected
          // generically here rather than typed per-column, since this table
          // renders both fold metrics and threshold-sweep rows.
          const values = row as unknown as Record<string, number | string>;
          // Only threshold-sweep rows have a "threshold" column at all, so
          // this is naturally a no-op for fold-metrics rows -- the row a
          // user actually bets on (see BuySignal's own default), not just a
          // sample among many in the sweep.
          const isDefaultThresholdRow =
            typeof values.threshold === "number" &&
            Math.abs(values.threshold - DEFAULT_BUY_THRESHOLD) < THRESHOLD_MATCH_EPSILON;
          return (
            <tr key={i} className={isDefaultThresholdRow ? "metrics-row-highlight" : undefined}>
              {columns.map((c) => {
                const value = values[c];
                // hit_rate answers "if I take a trade at this threshold, how
                // often am I supposed to make money" -- the one number here
                // a user actually acts on, so it's shown as a percentage
                // (not a raw 0-1 fraction) and given visual weight, not
                // buried as just another column.
                if (c === "hit_rate" && typeof value === "number") {
                  return (
                    <td className="trade-num" key={c} style={{ color: "var(--accent)", fontWeight: 700 }}>
                      {(value * 100).toFixed(1)}%
                      {isDefaultThresholdRow && <span className="threshold-highlight-badge">bet</span>}
                    </td>
                  );
                }
                return (
                  <td className="trade-num" key={c}>
                    {typeof value === "number"
                      ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
                      : value}
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function RunCard({ run }: { run: TrainingRunRecord }) {
  if (run.status === "failed") {
    return (
      <details className="view-card">
        <summary>
          <strong style={{ color: "var(--bad)" }}>Failed</strong> -- {formatTimestamp(run.started_at)}
          {" · "}
          {formatDuration(run.duration_seconds)}
        </summary>
        <div className="view-features">
          <p className="error">{run.error}</p>
        </div>
      </details>
    );
  }

  const holdout = run.holdout_metrics;

  return (
    <details className="view-card">
      <summary>
        <strong style={{ color: "var(--accent)" }}>{formatTimestamp(run.started_at)}</strong>
        {" · "}
        {formatDuration(run.duration_seconds)}
        {holdout && ` · ${(holdout.directional_accuracy * 100).toFixed(1)}% holdout accuracy`}
        {run.model_specs && (
          <div className="view-meta">{formatComposition(run.model_specs)}</div>
        )}
      </summary>
      <div className="view-features">
        <div className="view-metagrid">
          {run.train_tickers && (
            <div className="meta-item">
              <span className="meta-key">Trained on</span>
              <span className="meta-value">
                <CountWithTooltip items={run.train_tickers} label="tickers" />
              </span>
            </div>
          )}
          {run.holdout_tickers && (
            <div className="meta-item">
              <span className="meta-key">Held out</span>
              <span className="meta-value">
                <CountWithTooltip items={run.holdout_tickers} label="tickers" />
              </span>
            </div>
          )}
          {run.date_range && (
            <div className="meta-item">
              <span className="meta-key">Date range</span>
              <span className="meta-value">
                {run.date_range[0]} to {run.date_range[1]}
              </span>
            </div>
          )}
          {run.resolved_features && (
            <div className="meta-item">
              <span className="meta-key">Features</span>
              <span className="meta-value">
                <CountWithTooltip items={run.resolved_features} label="features" />
              </span>
            </div>
          )}
          {run.git_commit && (
            <div className="meta-item">
              <span className="meta-key">Commit</span>
              <span className="meta-value">{run.git_commit.slice(0, 7)}</span>
            </div>
          )}
        </div>
        {run.fold_metrics && <MetricsTable rows={run.fold_metrics} />}
        {run.threshold_sweep && <MetricsTable rows={run.threshold_sweep} />}
      </div>
    </details>
  );
}

export default function RunHistory() {
  // Fetched once, not polled -- unlike current job status, a past run never
  // changes once recorded, so there's nothing to keep re-fetching for.
  const { data, error } = useFetchData<TrainingRunsResponse>(fetchTrainingRuns);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading run history...</p>;
  if (data.runs.length === 0) return <p className="muted">No runs recorded yet.</p>;

  return (
    <div>
      {data.runs.map((run) => (
        <RunCard run={run} key={run.run_id} />
      ))}
    </div>
  );
}
