import { useState } from "react";
import {
  fetchPaperBook,
  fetchTrainingRuns,
  rebuildPaperBook,
  replayPaperBook,
  type PaperBookDay,
  type PaperBookResponse,
  type PaperListStats,
  type PaperPickRow,
  type TrainingRunRecord,
} from "../api";
import { useFetchData } from "../useFetchData";
import { DataTable, NewsCell, ScoreCell, TickerCell, UsdCell } from "./DataTable";
import TogglePill from "./TogglePill";

type Kind = "fit" | "rank" | "both";

function listStats(rows: PaperPickRow[]): PaperListStats {
  const flagged = rows.filter((row) => row.news_blocks);
  const kept = rows.filter((row) => !row.news_blocks);
  const rets = kept.map((row) => row.session_return).filter((value): value is number => value !== null);
  const nScored = rets.length;
  const wins = rets.filter((value) => value > 0).length;
  const losses = rets.filter((value) => value < 0).length;
  const avg = nScored ? rets.reduce((sum, value) => sum + value, 0) / nScored : null;
  return {
    n: rows.length,
    n_scored: nScored,
    wins,
    losses,
    flats: nScored - wins - losses,
    hit_rate: nScored ? wins / nScored : null,
    avg,
    n_avoid: flagged.length,
    avg_ex_news: avg,
  };
}

function compound(avgs: number[]): number | null {
  if (avgs.length === 0) return null;
  return avgs.reduceRight((wealth, avg) => wealth * (1 + avg), 1) - 1;
}

function takeTop(rows: PaperPickRow[], topK: number | undefined): PaperPickRow[] {
  if (topK === undefined) return rows;
  return rows.filter((row) => row.rank <= topK);
}

function sliceBook(
  data: PaperBookResponse,
  kind: Kind,
  fitTopK: number | undefined,
  rankTopK: number | undefined,
): PaperBookResponse {
  const days = data.days.map((day) => {
    const fit = kind === "rank" ? [] : takeTop(day.fit, fitTopK);
    const rank = kind === "fit" ? [] : takeTop(day.rank, rankTopK);
    const fitStats = listStats(fit);
    const rankStats = listStats(rank);
    return {
      ...day,
      fit,
      rank,
      fit_avg: fitStats.avg,
      rank_avg: rankStats.avg,
      fit_stats: fitStats,
      rank_stats: rankStats,
    };
  });
  const fitRows = days.flatMap((day) => day.fit);
  const rankRows = days.flatMap((day) => day.rank);
  const fitAvgs = days.map((day) => day.fit_stats?.avg).filter((value): value is number => value !== null && value !== undefined);
  const rankAvgs = days.map((day) => day.rank_stats?.avg).filter((value): value is number => value !== null && value !== undefined);
  return {
    ...data,
    days,
    fit_compound: compound(fitAvgs),
    rank_compound: compound(rankAvgs),
    fit_days: fitAvgs.length,
    rank_days: rankAvgs.length,
    fit_stats: listStats(fitRows),
    rank_stats: listStats(rankRows),
    kind,
    n_picks: fitRows.length + rankRows.length,
  };
}

function formatRunLabel(startedAt: string, holdoutAccuracy: number | null): string {
  const when = new Date(startedAt).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  return holdoutAccuracy === null ? when : `${when} · ${(holdoutAccuracy * 100).toFixed(1)}% holdout`;
}

function formatDay(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

function Pct({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  const isUp = value >= 0;
  return (
    <span className={isUp ? "quote-diff-up" : "quote-diff-down"}>
      {isUp ? "▲" : "▼"} {(Math.abs(value) * 100).toFixed(2)}%
    </span>
  );
}

function StatsLine({ stats }: { stats: PaperListStats | undefined }) {
  if (!stats || stats.n_scored === 0) return null;
  const hit = stats.hit_rate !== null ? `${(stats.hit_rate * 100).toFixed(0)}% hit` : null;
  return (
    <>
      {` · ${stats.n_scored} scored · ${stats.wins}W/${stats.losses}L`}
      {hit ? ` · ${hit}` : ""}
      {" · avg "}
      <Pct value={stats.avg} />
      {stats.n_avoid ? ` · skipped ${stats.n_avoid} gap-down news` : ""}
    </>
  );
}

function ListTable({
  title,
  rows,
  stats,
  isRank,
}: {
  title: string;
  rows: PaperPickRow[];
  stats: PaperListStats | undefined;
  isRank: boolean;
}) {
  if (rows.length === 0) {
    return <p className="muted">{title}: no list that morning</p>;
  }
  return (
    <div>
      <p className="view-meta" style={{ marginBottom: "var(--space-2)" }}>
        <strong>{title}</strong>
        {` · ${rows.length} names`}
        <StatsLine stats={stats} />
      </p>
      <div style={{ overflowX: "auto" }}>
        <DataTable
          rows={rows}
          rowKey={(row) => `${title}-${row.rank}-${row.ticker}`}
          columns={[
            { key: "rank", header: "#", cell: (row) => row.rank },
            { key: "ticker", header: "Ticker", cell: (row) => <TickerCell ticker={row.ticker} /> },
            {
              key: "score",
              header: isRank ? "Score" : "Predicted",
              numeric: true,
              cell: (row) => <ScoreCell value={row.predicted} isRank={isRank} />,
            },
            { key: "open", header: "Open", numeric: true, cell: (row) => <UsdCell value={row.open_price} /> },
            { key: "close", header: "Close", numeric: true, cell: (row) => <UsdCell value={row.close_price} /> },
            { key: "session", header: "Session", numeric: true, cell: (row) => <Pct value={row.session_return} /> },
            { key: "news", header: "News", cell: (row) => <NewsCell flag={row.news_flag} blocks={row.news_blocks} /> },
          ]}
        />
      </div>
    </div>
  );
}

function shortfallNote(asked: number | undefined, got: number, label: "Rank" | "Fit"): string | null {
  if (asked === undefined || got === 0 || got >= asked) return null;
  return `${label} had ${got} that morning`;
}

function DayCard({
  day,
  kind,
  rankAsked,
  fitAsked,
  modelLabel,
}: {
  day: PaperBookDay;
  kind: Kind;
  rankAsked: number | undefined;
  fitAsked: number | undefined;
  modelLabel?: string;
}) {
  const rankNote = kind !== "fit" ? shortfallNote(rankAsked, day.rank.length, "Rank") : null;
  const fitNote = kind !== "rank" ? shortfallNote(fitAsked, day.fit.length, "Fit") : null;
  return (
    <details className="view-card">
      <summary>
        <strong style={{ color: "var(--accent)" }}>{formatDay(day.as_of)}</strong>{" "}
        <span className="view-meta">
          {day.scan_id ? "replay · " : "live · "}
          {modelLabel ? <span className="muted">{modelLabel} · </span> : null}
          {kind !== "fit" && day.rank.length > 0 && (
            <>
              Rank {day.rank.length} <Pct value={day.rank_avg} />
            </>
          )}
          {kind === "both" && day.fit.length > 0 && day.rank.length > 0 && " · "}
          {kind !== "rank" && day.fit.length > 0 && (
            <>
              Fit {day.fit.length} <Pct value={day.fit_avg} />
            </>
          )}
          {(rankNote || fitNote) && (
            <>
              {" · "}
              <span className="muted">{[rankNote, fitNote].filter(Boolean).join(" · ")}</span>
            </>
          )}
        </span>
      </summary>
      <div style={{ display: "grid", gap: "var(--space-4)", marginTop: "var(--space-3)" }}>
        {kind !== "rank" && (
          <ListTable title="Fit" rows={day.fit} stats={day.fit_stats} isRank={false} />
        )}
        {kind !== "fit" && (
          <ListTable title="Rank" rows={day.rank} stats={day.rank_stats} isRank={true} />
        )}
      </div>
    </details>
  );
}

function parseTop(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (trimmed === "") return undefined;
  const n = Number(trimmed);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  return Math.floor(n);
}

function TopField({
  label,
  enabled,
  onEnabled,
  value,
  onValue,
}: {
  label: string;
  enabled: boolean;
  onEnabled: (on: boolean) => void;
  value: string;
  onValue: (value: string) => void;
}) {
  return (
    <TogglePill
      on={enabled}
      onToggle={() => onEnabled(!enabled)}
      extra={
        <input
          className="form-input slice-input"
          type="number"
          min={1}
          step={1}
          inputMode="numeric"
          placeholder="all"
          disabled={!enabled}
          value={value}
          onChange={(event) => onValue(event.target.value)}
          aria-label={`${label} top`}
        />
      }
    >
      {label}
    </TogglePill>
  );
}

export default function WhatIf() {
  const [rankOn, setRankOn] = useState(true);
  const [fitOn, setFitOn] = useState(true);
  const [rankText, setRankText] = useState("5");
  const [fitText, setFitText] = useState("5");
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [rebuildError, setRebuildError] = useState<string | null>(null);
  const [replayDay, setReplayDay] = useState("");
  const [replayModel, setReplayModel] = useState("live");
  const { data: raw, error } = useFetchData(() => fetchPaperBook("both"), { deps: [refresh] });
  const { data: trainingRuns } = useFetchData(fetchTrainingRuns, { deps: [] });

  if (error) return <p className="error">{error}</p>;
  if (!raw) return <p className="muted">Loading…</p>;

  const rankAsked = rankOn ? parseTop(rankText) : undefined;
  const fitAsked = fitOn ? parseTop(fitText) : undefined;
  const kind: Kind = rankOn && fitOn ? "both" : rankOn ? "rank" : "fit";
  const data = sliceBook(raw, kind, fitOn ? fitAsked : undefined, rankOn ? rankAsked : undefined);

  return (
    <div>
      <div className="slice-bar">
        <TopField label="Rank" enabled={rankOn} onEnabled={setRankOn} value={rankText} onValue={setRankText} />
        <TopField label="Fit" enabled={fitOn} onEnabled={setFitOn} value={fitText} onValue={setFitText} />
        <button
          type="button"
          className="icon-btn"
          disabled={busy}
          title="Fill Open→Close after the session"
          aria-label="Update What if after the close"
          onClick={async () => {
            setBusy(true);
            setRebuildError(null);
            try {
              await rebuildPaperBook();
              setRefresh((n) => n + 1);
            } catch (err) {
              setRebuildError(String(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Updating…" : "↻"}
        </button>
        {rebuildError && <span className="error">{rebuildError}</span>}
      </div>
      <div className="slice-bar">
        <input
          className="form-input"
          type="date"
          value={replayDay || raw.days[0]?.as_of || ""}
          onChange={(event) => setReplayDay(event.target.value)}
          aria-label="Replay day"
        />
        <select
          className="form-select"
          value={replayModel}
          onChange={(event) => setReplayModel(event.target.value)}
          aria-label="Replay model"
        >
          <option value="live">Live Fit pickle</option>
          {(trainingRuns?.runs ?? [])
            .filter((run: TrainingRunRecord) => run.has_archived_model)
            .map((run: TrainingRunRecord) => (
              <option key={run.run_id} value={run.run_id}>
                {formatRunLabel(run.started_at, run.holdout_metrics?.directional_accuracy ?? null)}
              </option>
            ))}
        </select>
        <button
          type="button"
          className="icon-btn"
          disabled={busy || !replayDay}
          onClick={async () => {
            setBusy(true);
            setRebuildError(null);
            try {
              await replayPaperBook(replayDay, replayModel === "live" ? null : replayModel);
              setRefresh((n) => n + 1);
            } catch (err) {
              setRebuildError(String(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Scoring…" : "Run"}
        </button>
      </div>
      {data.days.length === 0 || (!rankOn && !fitOn) ? (
        <p className="muted">{!rankOn && !fitOn ? "Turn on Rank or Fit." : "No morning lists yet."}</p>
      ) : (
        <>
          <div className="slice-result">
            {rankOn && (
              <div className="view-card slice-card">
                <div className="slice-card-kicker">
                  Rank · {rankAsked === undefined ? "all" : `top ${rankAsked}`} · {data.rank_days}d
                </div>
                <div className="slice-card-avg">
                  <Pct value={data.rank_compound} /> compound
                </div>
                <p className="view-meta" style={{ marginTop: 4 }}>
                  {data.rank_stats?.n ?? 0} names
                  <StatsLine stats={data.rank_stats} />
                </p>
              </div>
            )}
            {fitOn && (
              <div className="view-card slice-card">
                <div className="slice-card-kicker">
                  Fit · {fitAsked === undefined ? "all" : `top ${fitAsked}`} · {data.fit_days}d
                </div>
                <div className="slice-card-avg">
                  <Pct value={data.fit_compound} /> compound
                </div>
                <p className="view-meta" style={{ marginTop: 4 }}>
                  {data.fit_stats?.n ?? 0} names
                  <StatsLine stats={data.fit_stats} />
                </p>
              </div>
            )}
          </div>
          {data.days.map((day) => (
            <DayCard
              day={day}
              kind={kind}
              rankAsked={rankAsked}
              fitAsked={fitAsked}
              modelLabel={
                day.model_run_id
                  ? (() => {
                      const run = (trainingRuns?.runs ?? []).find((item) => item.run_id === day.model_run_id);
                      return run
                        ? formatRunLabel(run.started_at, run.holdout_metrics?.directional_accuracy ?? null)
                        : day.model_run_id.slice(0, 8);
                    })()
                  : day.scan_id
                    ? "Live Fit pickle"
                    : undefined
              }
              key={`${day.as_of}-${day.scan_id || "live"}`}
            />
          ))}
        </>
      )}
    </div>
  );
}
