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
import { isCashSessionToday, sessionHasClosed } from "../session";
import { useFetchData } from "../useFetchData";
import { useQuotes } from "../useQuotes";
import { Diff } from "./Diff";
import FreshnessBadge from "./FreshnessBadge";

const LATEST_OPTION_VALUE = "";
const DEFAULT_THRESHOLD_PCT = DEFAULT_BUY_THRESHOLD * 100;
const NO_MODEL_SENTINEL = "";
const LOADING_PHRASES = ["Fetching this morning's quotes...", "Scoring tickers...", "Ranking picks..."];
const LOADING_PHRASE_INTERVAL_MS = 900;

function formatRunLabel(startedAt: string, holdoutAccuracy: number | null): string {
  const when = new Date(startedAt).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  return holdoutAccuracy === null ? when : `${when} · ${(holdoutAccuracy * 100).toFixed(1)}% holdout`;
}

function formatToday(): string {
  return new Date().toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

function newsCell(signal: { news_flag?: string | null; news_blocks?: boolean }) {
  if (!signal.news_flag) {
    return { className: "muted" as const, text: "—" };
  }
  return {
    className: signal.news_blocks ? ("quote-diff-down" as const) : ("quote-diff-up" as const),
    text: `${signal.news_blocks ? "skip" : "still buy"} · ${signal.news_flag}`,
  };
}

function MorningList({
  title,
  list,
  isRank,
  showLive,
  closed,
}: {
  title: string;
  list: BuySignalResponse | null;
  isRank: boolean;
  showLive: boolean;
  closed: boolean;
}) {
  const [open, setOpen] = useState(false);
  const tickers = list?.signals.map((s) => s.ticker) ?? [];
  const { quotes } = useQuotes(tickers, {
    enabled: showLive && open && tickers.length > 0,
    intervalMs: closed || tickers.length > 40 ? undefined : 60_000,
  });
  if (!list) return null;
  const noModel = list.skipped.some((s) => s.ticker === NO_MODEL_SENTINEL);
  if (noModel) return null;
  if (list.signals.length === 0 && list.scored_count === 0) return null;
  const liveLabel = closed ? "Close" : "Last";
  return (
    <details
      className="view-card morning-list"
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
    >
      <summary>
        <strong>{title}</strong>
        {` · ${list.as_of} · ${list.signals.length} · scored ${list.scored_count}`}
      </summary>
      {list.signals.length === 0 ? (
        <p className="muted">No tickers on this list.</p>
      ) : (
        <table className="trade-table" style={{ marginTop: "var(--space-2)" }}>
          <thead>
            <tr>
              <th>Ticker</th>
              <th className="trade-num">{isRank ? "Score" : "Predicted"}</th>
              <th className="trade-num">Open</th>
              {showLive && <th className="trade-num">{liveLabel}</th>}
              <th>News</th>
            </tr>
          </thead>
          <tbody>
            {list.signals.map((signal) => {
              const news = newsCell(signal);
              const last = quotes[signal.ticker]?.last;
              return (
                <tr key={`${title}-${signal.ticker}`}>
                  <td className="trade-ticker">{signal.ticker}</td>
                  <td className="trade-num">
                    {isRank
                      ? signal.predicted_return.toFixed(4)
                      : `${(signal.predicted_return * 100).toFixed(2)}%`}
                  </td>
                  <td className="trade-num">{formatUsd(signal.open_price)}</td>
                  {showLive && (
                    <td className="trade-num">
                      {last === undefined ? (
                        <span className="muted">—</span>
                      ) : (
                        <>
                          {formatUsd(last)}{" "}
                          <Diff
                            value={last - signal.open_price}
                            pct={signal.open_price ? (last - signal.open_price) / signal.open_price : null}
                          />
                        </>
                      )}
                    </td>
                  )}
                  <td className={news.className}>{news.text}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </details>
  );
}

export default function BuySignal() {
  const [thresholdPct, setThresholdPct] = useState(DEFAULT_THRESHOLD_PCT);
  const [rankList, setRankList] = useState<BuySignalResponse | null>(null);
  const [fitList, setFitList] = useState<BuySignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingPhraseIndex, setLoadingPhraseIndex] = useState(0);
  const [jobTick, setJobTick] = useState(0);
  const { data: universe } = useFetchData<UniverseResponse>(fetchUniverse);
  const { data: trainingRuns } = useFetchData<TrainingRunsResponse>(fetchTrainingRuns);
  const [modelRefreshCount, setModelRefreshCount] = useState(0);
  const { data: liveModel } = useFetchData<LiveModelResponse>(fetchLiveModel, {
    deps: [modelRefreshCount],
  });
  const { data: scanStatus } = useFetchData(fetchMorningScan, {
    deps: [jobTick],
    intervalMs: 2000,
  });
  const scanRunning = scanStatus?.status === "running";

  async function handleModelChange(runId: string) {
    if (runId === LATEST_OPTION_VALUE) {
      await resetLiveModel();
    } else {
      await setLiveModel(runId);
    }
    setModelRefreshCount((c) => c + 1);
  }

  useEffect(() => {
    if (!loading && !scanRunning) {
      setLoadingPhraseIndex(0);
      return;
    }
    const intervalId = setInterval(() => {
      setLoadingPhraseIndex((i) => (i + 1) % LOADING_PHRASES.length);
    }, LOADING_PHRASE_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [loading, scanRunning]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetchBuySignal(thresholdPct / 100, false, "rank"),
      fetchBuySignal(thresholdPct / 100, false, "fit"),
    ])
      .then(([rank, fit]) => {
        if (!cancelled) {
          setRankList(rank);
          setFitList(fit);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [thresholdPct, scanStatus?.status, scanStatus?.completed_at]);

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
        await setMorningJob(true);
        setJobTick((n) => n + 1);
      }
      const [rank, fit] = await Promise.all([
        fetchBuySignal(thresholdPct / 100, false, "rank"),
        fetchBuySignal(thresholdPct / 100, false, "fit"),
      ]);
      setRankList(rank);
      setFitList(fit);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      {universe && (
        <p className="muted" style={{ marginBottom: 8 }}>
          {formatToday()} · Sell by close · {universe.active_ticker_count.toLocaleString()} tickers scanned.
        </p>
      )}
      <FreshnessBadge />
      <div className="trading-scan-controls">
        {trainingRuns && liveModel && (
          <div className="form-row" style={{ alignItems: "center", marginTop: 0 }}>
            <label className="muted" style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ minWidth: 88 }}>Model</span>
              <select
                className="form-select"
                value={liveModel.selected_run_id ?? LATEST_OPTION_VALUE}
                onChange={(e) => handleModelChange(e.target.value)}
                style={{ padding: "10px 8px", fontSize: "var(--text-body)" }}
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
        <div className="form-row" style={{ alignItems: "center", marginTop: 0 }}>
          <label className="muted" style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ minWidth: 88 }}>Threshold</span>
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
          <button className="btn-hero" onClick={handleCheck} disabled={loading || scanRunning}>
            {loading || scanRunning ? LOADING_PHRASES[loadingPhraseIndex] : "Check this morning's prices"}
          </button>
        </div>
      </div>
      <MorningTrigger jobNonce={jobTick} />

      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
      {scanRunning && (
        <p className="muted" style={{ marginTop: 8 }}>
          Getting this morning's prices…
        </p>
      )}
      {scanStatus?.status === "failed" && (
        <p className="error" style={{ marginTop: 8 }}>
          {scanStatus.error || "Morning run failed"}
        </p>
      )}

      {!error && (rankList || fitList) && (
        <MorningLists rankList={rankList} fitList={fitList} />
      )}
    </div>
  );
}

function MorningLists({
  rankList,
  fitList,
}: {
  rankList: BuySignalResponse | null;
  fitList: BuySignalResponse | null;
}) {
  const closed = sessionHasClosed();
  const rankToday = isCashSessionToday(rankList?.as_of);
  const fitToday = isCashSessionToday(fitList?.as_of);
  const headerAsOf = rankToday ? rankList?.as_of : fitToday ? fitList?.as_of : rankList?.as_of ?? fitList?.as_of;
  return (
    <details className="view-card morning-fold" style={{ marginTop: 12 }} open>
      <summary>
        <strong>This morning</strong>
        {headerAsOf ? ` · ${headerAsOf}` : ""}
      </summary>
      <div className="morning-lists">
        <MorningList
          title="Rank"
          list={rankList}
          isRank={true}
          showLive={rankToday}
          closed={closed}
        />
        <MorningList
          title="Fit 0.5%"
          list={fitList}
          isRank={false}
          showLive={fitToday}
          closed={closed}
        />
      </div>
    </details>
  );
}

function MorningTrigger({
  jobNonce = 0,
}: {
  jobNonce?: number;
}) {
  const { data: job, error: jobError } = useFetchData(fetchMorningJob, { deps: [jobNonce] });

  async function toggleJob(enabled: boolean) {
    await setMorningJob(enabled);
  }

  return (
    <div className="meta-row" style={{ marginTop: "var(--space-3)" }}>
      <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          type="checkbox"
          checked={job?.enabled !== false}
          onChange={(event) => toggleJob(event.target.checked)}
        />
        If I don't click, start at 8:31 anyway
      </label>
      {jobError && <span className="error">{jobError}</span>}
    </div>
  );
}
