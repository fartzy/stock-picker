import { useEffect, useState } from "react";
import {
  DEFAULT_BUY_THRESHOLD,
  fetchBuySignal,
  fetchLiveModel,
  fetchMorningJob,
  fetchMorningScan,
  fetchTrainingRuns,
  fetchUniverse,
  resetLiveModel,
  runMorningScan,
  setLiveModel,
  setMorningJob,
  type BuySignalResponse,
  type LiveModelResponse,
  type TrainingRunsResponse,
  type UniverseResponse,
} from "../api";
import { formatUsd } from "../format";
import { useFetchData } from "../useFetchData";
import FreshnessBadge from "./FreshnessBadge";

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
  const [listKind, setListKind] = useState<"rank" | "fit">("rank");
  const [jobTick, setJobTick] = useState(0);
  const [autoRunOff, setAutoRunOff] = useState(false);
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

  async function waitForScan(): Promise<void> {
    const deadline = Date.now() + 8 * 60 * 1000;
    while (Date.now() < deadline) {
      const scan = await fetchMorningScan();
      if (scan.status === "completed") return;
      if (scan.status === "failed") {
        throw new Error(scan.error || "morning scan failed");
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    throw new Error("morning scan is still running");
  }

  async function handleCheck() {
    setLoading(true);
    setError(null);
    try {
      const existing = await fetchMorningScan();
      if (existing.status === "running") {
        setError("Job already running — getting this morning's prices.");
        await waitForScan();
      } else {
        setAutoRunOff(true);
        await setMorningJob(false);
        setJobTick((n) => n + 1);
        try {
          await runMorningScan();
        } catch (err) {
          const message = String(err);
          if (!message.includes("409") && !/already/i.test(message)) {
            throw err;
          }
          setError("Job already running — getting this morning's prices.");
        }
        await waitForScan();
      }
      setData(await fetchBuySignal(thresholdPct / 100, false, listKind));
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const displayed = data;
  const showingCache = Boolean(displayed?.cached);
  const noModel = displayed?.skipped.some((s) => s.ticker === NO_MODEL_SENTINEL) ?? false;

  return (
    <div>
      {universe && (
        <p className="muted" style={{ marginBottom: 8 }}>
          {formatToday()} · Sell by close · {universe.active_ticker_count.toLocaleString()} tickers scanned.
        </p>
      )}
      <FreshnessBadge />
      <div className="list-toggle" role="group" aria-label="Pick list">
        <button
          type="button"
          className={listKind === "rank" ? "active" : ""}
          onClick={() => {
            setListKind("rank");
            setData(null);
          }}
        >
          Rank
        </button>
        <button
          type="button"
          className={listKind === "fit" ? "active" : ""}
          onClick={() => {
            setListKind("fit");
            setData(null);
          }}
        >
          Fit 0.5%
        </button>
      </div>
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
      <MorningTrigger jobNonce={jobTick} forceOff={autoRunOff} />

      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
      {showingCache && displayed && (
        <p className="muted" style={{ marginTop: 8 }}>
          Loaded this morning's saved scan ({displayed.as_of}).
        </p>
      )}

      {displayed && !error && noModel && (
        <p className="muted" style={{ marginTop: 12 }}>
          No trained model yet. Train one on the Models tab first.
        </p>
      )}

      {displayed && !error && !noModel && (
        <div style={{ marginTop: 12 }}>
          {displayed.top_drivers.length > 0 && (
            <p className="muted">
              Top drivers: {displayed.top_drivers.map((d) => `${d.feature} (${d.importance.toFixed(1)}%)`).join(", ")}
            </p>
          )}
          {displayed.signals.length === 0 ? (
            <p className="muted">No tickers cleared the {thresholdPct}% threshold this morning.</p>
          ) : (
            <table className="trade-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="trade-num">{listKind === "rank" ? "Rank score" : "Predicted return"}</th>
                  <th className="trade-num">Open price</th>
                  <th>News</th>
                </tr>
              </thead>
              <tbody>
                {displayed.signals.map((signal) => (
                  <tr key={signal.ticker}>
                    <td className="trade-ticker">{signal.ticker}</td>
                    <td className="trade-num">
                      {listKind === "rank"
                        ? signal.predicted_return.toFixed(4)
                        : `${(signal.predicted_return * 100).toFixed(2)}%`}
                    </td>
                    <td className="trade-num">{formatUsd(signal.open_price)}</td>
                    <td className={signal.news_flag ? "quote-diff-down" : "muted"}>
                      {signal.news_flag ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ marginTop: 8 }}>
            Scored {displayed.scored_count} of {displayed.scored_count + displayed.skipped.length} tickers ·{" "}
            <span title={displayed.skipped.map((s) => `${s.ticker}: ${s.reason}`).join("\n")}>
              {displayed.skipped.length} skipped
            </span>
          </p>
        </div>
      )}
    </div>
  );
}


function MorningTrigger({
  jobNonce = 0,
  forceOff = false,
}: {
  onDone?: () => void;
  jobNonce?: number;
  forceOff?: boolean;
}) {
  const { data: job, error: jobError } = useFetchData(fetchMorningJob, { deps: [jobNonce] });
  const [scanTick, setScanTick] = useState(0);
  const [scan, setScan] = useState<import("../api").MorningScanStatus | null>(null);
  const running = scan?.status === "running";

  useEffect(() => {
    let cancelled = false;
    fetchMorningScan()
      .then((result) => {
        if (!cancelled) setScan(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [scanTick]);

  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => {
      fetchMorningScan()
        .then(setScan)
        .catch(() => undefined);
    }, 2000);
    return () => clearInterval(id);
  }, [running]);

  async function toggleJob(enabled: boolean) {
    await setMorningJob(enabled);
    setScanTick((n) => n + 1);
  }

  return (
    <div className="meta-row" style={{ marginTop: "var(--space-3)" }}>
      <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          type="checkbox"
          checked={!forceOff && job?.enabled !== false}
          onChange={(event) => toggleJob(event.target.checked)}
        />
        If I don't click, start at 8:32 anyway
      </label>
      {running && <span className="view-meta">Getting this morning's prices…</span>}
      {scan?.status === "failed" && (
        <span className="error">{scan.error || "Couldn't get this morning's prices"}</span>
      )}
      {jobError && <span className="error">{jobError}</span>}
    </div>
  );
}
