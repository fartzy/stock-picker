import { useEffect, useState } from "react";
import {
  DEFAULT_BUY_THRESHOLD,
  fetchBuySignal,
  fetchLiveModel,
  fetchTrainingRuns,
  fetchUniverse,
  resetLiveModel,
  setLiveModel,
  type BuySignalResponse,
  type LiveModelResponse,
  type TrainingRunsResponse,
  type UniverseResponse,
} from "../api";
import { formatUsd } from "../format";
import { useFetchData } from "../useFetchData";

// "Latest" isn't a real run_id -- this <option>'s value means "clear the
// explicit selection," resolved via resetLiveModel() rather than setLiveModel().
const LATEST_OPTION_VALUE = "";

function formatRunLabel(startedAt: string, holdoutAccuracy: number | null): string {
  const when = new Date(startedAt).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  return holdoutAccuracy === null ? when : `${when} · ${(holdoutAccuracy * 100).toFixed(1)}% holdout`;
}

// Percent, not fraction -- shown as a plain "%" input.
const DEFAULT_THRESHOLD_PCT = DEFAULT_BUY_THRESHOLD * 100;

// Sentinel the backend uses for skipped[] when no model is trained yet at
// all, rather than "nothing cleared the bar today" -- see buy_signal.py.
// Deliberately not a valid ticker shape, so it can never collide with a
// real skipped ticker (an earlier "ALL" sentinel collided with Allstate's
// actual ticker symbol).
const NO_MODEL_SENTINEL = "";

// The real request is one round trip (live quotes -> score -> rank), not
// discrete steps -- these cycle purely to make a several-second wait feel
// alive and give a rough sense of what's happening, not to report genuine
// backend progress.
const LOADING_PHRASES = ["Fetching this morning's quotes...", "Scoring tickers...", "Ranking picks..."];
const LOADING_PHRASE_INTERVAL_MS = 900;

function formatToday(): string {
  return new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

export default function BuySignal() {
  const [thresholdPct, setThresholdPct] = useState(DEFAULT_THRESHOLD_PCT);
  const [data, setData] = useState<BuySignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingPhraseIndex, setLoadingPhraseIndex] = useState(0);
  // Cheap (a parquet read, no live quotes) so this shows up front, before
  // the user ever clicks -- otherwise the scan's actual breadth (every
  // ticker ever tracked, not some smaller subset) stays invisible until
  // after a full live run reveals it via scored_count/skipped.
  const { data: universe } = useFetchData<UniverseResponse>(fetchUniverse);
  const { data: trainingRuns } = useFetchData<TrainingRunsResponse>(fetchTrainingRuns);
  const [modelRefreshCount, setModelRefreshCount] = useState(0);
  const { data: liveModel } = useFetchData<LiveModelResponse>(fetchLiveModel, { deps: [modelRefreshCount] });

  async function handleModelChange(runId: string) {
    if (runId === LATEST_OPTION_VALUE) {
      await resetLiveModel();
    } else {
      await setLiveModel(runId);
    }
    setModelRefreshCount((c) => c + 1);
  }

  useEffect(() => {
    if (!loading) {
      setLoadingPhraseIndex(0);
      return;
    }
    const intervalId = setInterval(() => {
      setLoadingPhraseIndex((i) => (i + 1) % LOADING_PHRASES.length);
    }, LOADING_PHRASE_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [loading]);

  async function handleCheck() {
    setLoading(true);
    setError(null);
    try {
      setData(await fetchBuySignal(thresholdPct / 100));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const noModel = data?.skipped.some((s) => s.ticker === NO_MODEL_SENTINEL) ?? false;

  return (
    <div>
      {universe && (
        <p className="muted" style={{ marginBottom: 8 }}>
          {formatToday()} · Sell by close · {universe.active_ticker_count.toLocaleString()} tickers scanned.
        </p>
      )}
      {trainingRuns && liveModel && (
        <div className="form-row" style={{ alignItems: "center", marginTop: 0 }}>
          <label className="muted" style={{ display: "flex", alignItems: "center", gap: 10 }}>
            Model
            <select
              className="form-select"
              value={liveModel.selected_run_id ?? LATEST_OPTION_VALUE}
              onChange={(e) => handleModelChange(e.target.value)}
            >
              <option value={LATEST_OPTION_VALUE}>Latest</option>
              {trainingRuns.runs
                .filter((run) => run.has_archived_model)
                .map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {formatRunLabel(run.started_at, run.holdout_metrics?.directional_accuracy ?? null)}
                  </option>
                ))}
            </select>
          </label>
        </div>
      )}
      <div className="form-row" style={{ alignItems: "center" }}>
        <label className="muted" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          Threshold
          <input
            className="form-input"
            type="number"
            step="0.1"
            min="0"
            value={thresholdPct}
            onChange={(e) => setThresholdPct(Number(e.target.value))}
            style={{ width: 70, padding: "10px 8px", fontSize: "var(--text-body)" }}
          />
          <span>%</span>
        </label>
        <button className="btn-hero" onClick={handleCheck} disabled={loading}>
          {loading ? LOADING_PHRASES[loadingPhraseIndex] : "Check this morning's prices"}
        </button>
      </div>

      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}

      {data && !error && noModel && (
        <p className="muted" style={{ marginTop: 12 }}>
          No trained model yet. Train one on the Models tab first.
        </p>
      )}

      {data && !error && !noModel && (
        <div style={{ marginTop: 12 }}>
          {data.top_drivers.length > 0 && (
            <p className="muted">
              Top drivers: {data.top_drivers.map((d) => `${d.feature} (${d.importance.toFixed(1)}%)`).join(", ")}
            </p>
          )}
          {data.signals.length === 0 ? (
            <p className="muted">No tickers cleared the {thresholdPct}% threshold this morning.</p>
          ) : (
            <table className="trade-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="trade-num">Predicted return</th>
                  <th className="trade-num">Open price</th>
                </tr>
              </thead>
              <tbody>
                {data.signals.map((signal) => (
                  <tr key={signal.ticker}>
                    <td className="trade-ticker">{signal.ticker}</td>
                    <td className="trade-num">{(signal.predicted_return * 100).toFixed(2)}%</td>
                    <td className="trade-num">{formatUsd(signal.open_price)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ marginTop: 8 }}>
            Scored {data.scored_count} of {data.scored_count + data.skipped.length} tickers ·{" "}
            <span title={data.skipped.map((s) => `${s.ticker}: ${s.reason}`).join("\n")}>
              {data.skipped.length} skipped
            </span>
          </p>
        </div>
      )}
    </div>
  );
}
